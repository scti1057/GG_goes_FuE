import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from std_srvs.srv import Trigger
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs_msgs.msg import Keypoints


@dataclass
class Candidate:
    cid: int
    xy: np.ndarray              # (2,)
    desc: Optional[np.ndarray]  # (D,)
    count: int
    score_sum: float


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class ReferenceManagerNode(Node):
    def __init__(self):
        super().__init__('reference_manager_node')

        # Params
        self.declare_parameter('keypoints_topic', '/ibvs/keypoints')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')

        self.declare_parameter('init_duration_sec', 5.0)
        self.declare_parameter('ref_top_k', 300)

        self.declare_parameter('desc_match_threshold', 0.85)  # cosine
        self.declare_parameter('max_px_dist', 4.0)            # pixel gate (robot still)
        self.declare_parameter('desc_ema_alpha', 0.25)
        self.declare_parameter('reference_republish_hz', 2.0)

        self.declare_parameter('save_overlay_path', '/tmp/ibvs_reference_overlay.png')
        self.declare_parameter('save_npz_path', '')           # optional: /tmp/reference.npz
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('debug_topic', '/ibvs/debug/reference_candidates_image')

        # State
        self.bridge = CvBridge()
        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_img_header: Optional[Header] = None

        self.candidates: Dict[int, Candidate] = {}
        self.next_cid: int = 0

        self.capturing = False
        self.capture_t0 = 0.0

        # Reference
        self.reference_ready = False
        self.ref_xy: Optional[np.ndarray] = None    # (K,2)
        self.ref_desc: Optional[np.ndarray] = None  # (K,D) or None
        self.ref_counts: Optional[np.ndarray] = None
        self.last_ref_pub_t: float = 0.0

        # QoS: latched-like publisher for ref + init_done
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.init_done_pub = self.create_publisher(Bool, '/ibvs/init_done', latched_qos)
        self.ref_pub = self.create_publisher(Keypoints, '/ibvs/reference/keypoints', latched_qos)
        self.debug_pub = None
        if bool(self.get_parameter('debug_mode').value):
            self.debug_pub = self.create_publisher(Image, self.get_parameter('debug_topic').value, 10)

        # Subscribers
        kp_topic = self.get_parameter('keypoints_topic').value
        img_topic = self.get_parameter('image_topic').value
        self.kp_sub = self.create_subscription(Keypoints, kp_topic, self.on_keypoints, qos_profile_sensor_data)
        self.img_sub = self.create_subscription(Image, img_topic, self.on_image, qos_profile_sensor_data)

        # Service
        self.srv = self.create_service(Trigger, '/ibvs/reference/start_capture', self.on_start_capture)

        # Timer
        self.timer = self.create_timer(0.05, self.on_timer)

        self.publish_init_done(False)
        self.get_logger().info(f"Ready. keypoints_topic={kp_topic} image_topic={img_topic}")
        if self.debug_pub is not None:
            self.get_logger().info(f"Debug overlay topic={self.get_parameter('debug_topic').value}")

    def publish_init_done(self, v: bool):
        msg = Bool()
        msg.data = bool(v)
        self.init_done_pub.publish(msg)

    def on_image(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_img_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"image convert failed: {e}")

    def on_start_capture(self, req: Trigger.Request, resp: Trigger.Response):
        if self.capturing:
            resp.success = False
            resp.message = "Capture already running."
            return resp

        # reset
        self.candidates.clear()
        self.next_cid = 0
        self.reference_ready = False
        self.ref_xy = None
        self.ref_desc = None
        self.ref_counts = None
        self.last_ref_pub_t = 0.0

        self.capturing = True
        self.capture_t0 = time.time()
        self.publish_init_done(False)

        dur = float(self.get_parameter('init_duration_sec').value)
        topk = int(self.get_parameter('ref_top_k').value)
        resp.success = True
        resp.message = f"Started capture for {dur:.2f}s, will keep top {topk}."
        self.get_logger().info(resp.message)
        return resp

    def on_timer(self):
        if not self.capturing:
            self.publish_reference()
            return

        dur = float(self.get_parameter('init_duration_sec').value)
        if (time.time() - self.capture_t0) >= dur:
            self.finish_capture()

    def publish_reference(self, force: bool = False):
        if not self.reference_ready or self.ref_xy is None:
            return

        hz = float(self.get_parameter('reference_republish_hz').value)
        now = time.time()
        if not force:
            if hz <= 0.0:
                return
            period = 1.0 / hz
            if (now - self.last_ref_pub_t) < period:
                return

        if self.latest_img_header is not None:
            header = self.latest_img_header
        else:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = ""

        D = int(self.ref_desc.shape[1]) if self.ref_desc is not None else 0

        ref_msg = Keypoints()
        ref_msg.header = header
        ref_msg.xy = self.ref_xy.reshape(-1).tolist()
        ref_msg.descriptor_dim = D
        if self.ref_desc is None:
            ref_msg.descriptors = []
        else:
            ref_msg.descriptors = self.ref_desc.reshape(-1).tolist()
        ref_msg.scores = []

        self.ref_pub.publish(ref_msg)
        self.last_ref_pub_t = now

    def on_keypoints(self, msg: Keypoints):
        if not self.capturing:
            return

        xy = np.array(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn("xy length not even; skipping frame")
            return
        kpts = xy.reshape(-1, 2)
        n = kpts.shape[0]
        if n == 0:
            return

        D = int(msg.descriptor_dim)
        desc = None
        if D > 0 and len(msg.descriptors) == n * D:
            desc = np.array(msg.descriptors, dtype=np.float32).reshape(n, D)

        if len(msg.scores) == n:
            scores = np.array(msg.scores, dtype=np.float32)
        else:
            scores = np.zeros((n,), dtype=np.float32)

        max_px = float(self.get_parameter('max_px_dist').value)
        thr = float(self.get_parameter('desc_match_threshold').value)
        alpha = float(self.get_parameter('desc_ema_alpha').value)

        # spatial grid of existing candidates (for fast nearby lookup)
        cell = max(1.0, max_px)
        grid: Dict[Tuple[int, int], List[int]] = {}
        for cid, c in self.candidates.items():
            cx = int(c.xy[0] // cell)
            cy = int(c.xy[1] // cell)
            grid.setdefault((cx, cy), []).append(cid)

        def nearby_candidate_ids(x: float, y: float) -> List[int]:
            cx = int(x // cell)
            cy = int(y // cell)
            ids: List[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ids.extend(grid.get((cx + dx, cy + dy), []))
            return ids

        for i in range(n):
            x, y = float(kpts[i, 0]), float(kpts[i, 1])
            d_i = desc[i] if desc is not None else None
            s_i = float(scores[i])

            best_cid = None
            best_sim = -1.0
            best_dist = 1e9

            for cid in nearby_candidate_ids(x, y):
                c = self.candidates[cid]
                dx = float(c.xy[0] - x)
                dy = float(c.xy[1] - y)
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > max_px:
                    continue

                if d_i is None or c.desc is None:
                    sim = 1.0  # pixel-only match
                else:
                    sim = _cosine_sim(c.desc, d_i)

                if sim > best_sim or (abs(sim - best_sim) < 1e-6 and dist < best_dist):
                    best_sim = sim
                    best_dist = dist
                    best_cid = cid

            matched = False
            if best_cid is not None:
                matched = True if desc is None else (best_sim >= thr)

            if matched and best_cid is not None:
                c = self.candidates[best_cid]
                c.count += 1
                c.xy = np.array([x, y], dtype=np.float32)
                c.score_sum += s_i
                if d_i is not None and c.desc is not None:
                    c.desc = (1.0 - alpha) * c.desc + alpha * d_i
            else:
                cid = self.next_cid
                self.next_cid += 1
                self.candidates[cid] = Candidate(
                    cid=cid,
                    xy=np.array([x, y], dtype=np.float32),
                    desc=(d_i.copy() if d_i is not None else None),
                    count=1,
                    score_sum=s_i,
                )

        self.publish_debug_overlay(msg.header)

    def publish_debug_overlay(self, header: Header):
        if self.debug_pub is None or self.latest_bgr is None:
            return

        dbg = self.latest_bgr.copy()
        for c in self.candidates.values():
            x, y = int(c.xy[0]), int(c.xy[1])
            cv2.circle(dbg, (x, y), 2, (0, 255, 255), -1)

        if self.ref_xy is not None:
            for x, y in self.ref_xy:
                cv2.circle(dbg, (int(x), int(y)), 3, (0, 0, 255), -1)

        cv2.putText(
            dbg,
            f"capturing={self.capturing} candidates={len(self.candidates)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
        msg.header = header
        self.debug_pub.publish(msg)

    def finish_capture(self):
        self.capturing = False

        if len(self.candidates) == 0:
            self.get_logger().warn("Capture finished but no candidates collected.")
            self.publish_init_done(False)
            return

        topk = int(self.get_parameter('ref_top_k').value)
        cand_list = sorted(self.candidates.values(), key=lambda c: c.count, reverse=True)
        keep = cand_list[: min(topk, len(cand_list))]
        K = len(keep)

        self.ref_xy = np.stack([c.xy for c in keep], axis=0).astype(np.float32)
        self.ref_counts = np.array([c.count for c in keep], dtype=np.int32)

        if keep[0].desc is not None:
            self.ref_desc = np.stack([c.desc for c in keep], axis=0).astype(np.float32)  # type: ignore
        else:
            self.ref_desc = None

        self.reference_ready = True

        # Publish once immediately, then keep republishing from timer.
        self.publish_reference(force=True)

        # Save overlay
        save_path = str(self.get_parameter('save_overlay_path').value)
        if self.latest_bgr is not None and save_path:
            img = self.latest_bgr.copy()
            for x, y in self.ref_xy:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.imwrite(save_path, img)
            self.get_logger().info(f"Saved reference overlay: {save_path}")
            if self.debug_pub is not None:
                dbg_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                if self.latest_img_header is not None:
                    dbg_msg.header = self.latest_img_header
                self.debug_pub.publish(dbg_msg)

        # Optional npz
        npz_path = str(self.get_parameter('save_npz_path').value)
        if npz_path:
            np.savez(
                npz_path,
                ref_xy=self.ref_xy,
                ref_desc=(self.ref_desc if self.ref_desc is not None else np.zeros((0, 0), np.float32)),
                ref_counts=self.ref_counts,
            )
            self.get_logger().info(f"Saved reference npz: {npz_path}")

        self.publish_init_done(True)
        self.get_logger().info(f"Capture done. candidates={len(self.candidates)} kept={K} best_count={keep[0].count}")

def main():
    rclpy.init()
    node = ReferenceManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
