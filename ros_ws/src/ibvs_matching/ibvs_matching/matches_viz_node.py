import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from typing import Any

from ibvs_msgs.msg import Keypoints, Matches

class MatchesVizNode(Node):
    def __init__(self):
        super().__init__('matches_viz_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('matches_topic', '/ibvs/matches')
        self.declare_parameter('reference_topic', '/ibvs/reference/keypoints')
        self.declare_parameter('controller_cmd_topic', '/cartesian_twist_passthrough_controller/cmd_vel')
        self.declare_parameter('output_topic', '/ibvs/debug/matches_image')
        self.declare_parameter('miss_max', 10)
        self.declare_parameter('radius', 3)
        self.declare_parameter('draw_reference_points', True)
        self.declare_parameter('draw_reference_links', True)
        self.declare_parameter('max_reference_draw', 400)
        self.declare_parameter('fps_ema_alpha', 0.15)
        self.declare_parameter('cmd_rate_window_sec', 1.0)

        self.bridge = CvBridge()
        self.last_matches = None  # type: Matches | None
        self.ref_xy = None  # type: np.ndarray | None
        self.fps_ema = 0.0
        self.last_img_time_sec = -1.0
        self.cmd_times_sec = []  # type: list[float]

        # ref_id -> (x,y,missed)
        self.tracks = {}

        img_topic = self.get_parameter('image_topic').value
        m_topic = self.get_parameter('matches_topic').value
        ref_topic = self.get_parameter('reference_topic').value
        cmd_topic = self.get_parameter('controller_cmd_topic').value
        out_topic = self.get_parameter('output_topic').value

        self.image_topic_uses_compressed = self._topic_uses_compressed(img_topic)
        image_msg_type = CompressedImage if self.image_topic_uses_compressed else Image

        self.sub_img = self.create_subscription(image_msg_type, img_topic, self.on_image, qos_profile_sensor_data)
        self.sub_m = self.create_subscription(Matches, m_topic, self.on_matches, qos_profile_sensor_data)
        self.sub_ref = self.create_subscription(Keypoints, ref_topic, self.on_reference, 10)
        self.sub_cmd = self.create_subscription(Twist, cmd_topic, self.on_cmd_vel, qos_profile_sensor_data)

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(Image, out_topic, pub_qos)

        self.get_logger().info(
            f"Sub image:   {img_topic} "
            f"({'compressed' if self.image_topic_uses_compressed else 'raw'})"
        )
        self.get_logger().info(f"Sub matches: {m_topic}")
        self.get_logger().info(f"Sub ref:     {ref_topic}")
        self.get_logger().info(f"Sub cmd:     {cmd_topic}")
        self.get_logger().info(f"Pub overlay: {out_topic}")

    @staticmethod
    def _topic_uses_compressed(topic: str) -> bool:
        return topic.endswith('/compressed') or topic.endswith('/compressedDepth')

    def _decode_color_bgr(self, msg: Any) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            try:
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                bgr = None
            if bgr is None:
                bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError('compressed image decode returned None')
            return bgr
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def on_matches(self, msg: Matches):
        self.last_matches = msg

    def on_reference(self, msg: Keypoints):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn('Reference xy length not even; ignoring update.')
            return
        self.ref_xy = xy.reshape(-1, 2)

    def on_cmd_vel(self, _msg: Twist):
        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        self.cmd_times_sec.append(now_sec)
        self._prune_cmd_times(now_sec)

    def _prune_cmd_times(self, now_sec: float):
        window = max(0.2, float(self.get_parameter('cmd_rate_window_sec').value))
        cutoff = now_sec - window
        while self.cmd_times_sec and self.cmd_times_sec[0] < cutoff:
            self.cmd_times_sec.pop(0)

    def _update_fps(self, now_sec: float) -> float:
        fps_inst = 0.0
        if self.last_img_time_sec > 0.0:
            dt = now_sec - self.last_img_time_sec
            if dt > 1e-6:
                fps_inst = 1.0 / dt
        self.last_img_time_sec = now_sec

        alpha = float(np.clip(float(self.get_parameter('fps_ema_alpha').value), 0.0, 1.0))
        if self.fps_ema <= 0.0:
            self.fps_ema = fps_inst
        else:
            self.fps_ema = alpha * fps_inst + (1.0 - alpha) * self.fps_ema
        return self.fps_ema

    def _command_rate_hz(self, now_sec: float) -> float:
        self._prune_cmd_times(now_sec)
        window = max(0.2, float(self.get_parameter('cmd_rate_window_sec').value))
        if window <= 1e-6:
            return 0.0
        return float(len(self.cmd_times_sec)) / window

    def _draw_hud(self, out: np.ndarray, cam_fps_hz: float, cmd_rate_hz: float):
        x0, y0 = 10, 10
        w, h = 390, 88
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, f'Camera FPS: {cam_fps_hz:5.1f}', (x0 + 10, y0 + 28), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f'Cmd rate:   {cmd_rate_hz:5.1f} Hz', (x0 + 10, y0 + 55), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f'Tracks: {len(self.tracks):3d}', (x0 + 10, y0 + 79), font, 0.54, (200, 255, 200), 1, cv2.LINE_AA)

    def on_image(self, msg: Any):
        try:
            bgr = self._decode_color_bgr(msg)
        except Exception as e:
            self.get_logger().warn(f"img convert failed: {e}")
            return

        now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
        cam_fps_hz = self._update_fps(now_sec)
        cmd_rate_hz = self._command_rate_hz(now_sec)

        miss_max = int(self.get_parameter('miss_max').value)
        radius = int(self.get_parameter('radius').value)
        draw_ref_points = bool(self.get_parameter('draw_reference_points').value)
        draw_ref_links = bool(self.get_parameter('draw_reference_links').value)
        max_reference_draw = max(0, int(self.get_parameter('max_reference_draw').value))

        # 1) missed++ for all known tracks
        for rid in list(self.tracks.keys()):
            x, y, missed = self.tracks[rid]
            missed += 1
            if missed > miss_max:
                del self.tracks[rid]
            else:
                self.tracks[rid] = (x, y, missed)

        # 2) apply latest matches (only reference points!)
        if self.last_matches is not None:
            m = self.last_matches
            xy = np.array(m.xy, dtype=np.float32)
            if xy.size % 2 == 0:
                pts = xy.reshape(-1, 2)
                n = min(len(m.ref_id), pts.shape[0])
                for i in range(n):
                    rid = m.ref_id[i]
                    x, y = float(pts[i, 0]), float(pts[i, 1])
                    self.tracks[int(rid)] = (x, y, 0)  # reset missed

        # 3) draw tracks with fading (color intensity)
        out = bgr.copy()
        for rid, (x, y, missed) in self.tracks.items():
            intensity = max(0.0, 1.0 - (missed / float(miss_max)))
            g = int(255 * intensity)
            cv2.circle(out, (int(x), int(y)), radius, (0, g, 0), -1)

        if draw_ref_points and self.ref_xy is not None and self.ref_xy.shape[0] > 0:
            n_ref = int(self.ref_xy.shape[0])
            n_draw = min(n_ref, max_reference_draw) if max_reference_draw > 0 else n_ref
            for i in range(n_draw):
                rx, ry = self.ref_xy[i]
                cv2.circle(out, (int(rx), int(ry)), 2, (0, 0, 255), -1)

        if draw_ref_links and self.last_matches is not None and self.ref_xy is not None and self.ref_xy.shape[0] > 0:
            m = self.last_matches
            xy = np.asarray(m.xy, dtype=np.float32)
            n_xy = int(xy.size // 2)
            n = min(len(m.ref_id), n_xy)
            if n > 0:
                pts = xy.reshape(-1, 2)
                for i in range(n):
                    rid = int(m.ref_id[i])
                    if rid < 0 or rid >= self.ref_xy.shape[0]:
                        continue
                    cur = pts[i]
                    ref = self.ref_xy[rid]
                    p_cur = (int(cur[0]), int(cur[1]))
                    p_ref = (int(ref[0]), int(ref[1]))
                    cv2.circle(out, p_ref, 2, (0, 0, 255), -1)
                    cv2.circle(out, p_cur, radius, (0, 255, 0), -1)

        self._draw_hud(out, cam_fps_hz, cmd_rate_hz)

        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = MatchesVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
