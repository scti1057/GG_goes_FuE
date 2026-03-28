import cv2
import numpy as np
import rclpy
import struct
from cv_bridge import CvBridge
from ibvs_msgs.msg import Keypoints
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Image
from typing import Any

from ibvs_perception.detectors import create_detector


class KeypointNode(Node):
    def __init__(self):
        super().__init__('keypoint_node')

        # Topics
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('depth_topic', 'lis')
        self.declare_parameter('keypoints_topic', '/ibvs/keypoints')

        # Debug topics
        self.declare_parameter('debug_mode', True)
        self.declare_parameter('output_topic', '/ibvs/debug/keypoints_image')
        self.declare_parameter('binary_output_topic', '/ibvs/debug/near_mask')

        # Detector params
        self.declare_parameter('detector_type', 'sift')   # sift|akaze|orb|superpoint|aliked|xfeat
        self.declare_parameter('device', 'cpu')           # cpu (später cuda möglich)
        self.declare_parameter('max_num_keypoints', 1024) # for superpoint/aliked
        self.declare_parameter('top_k', 1024)             # for xfeat
        self.declare_parameter('xfeat_repo_dir', '')      # optional absolute path to accelerated_features repo
        self.declare_parameter('nfeatures', 800)          # for sift/orb

        # Depth ROI params
        self.declare_parameter('use_depth_roi', True)
        self.declare_parameter('depth_scale', 0.001)      # uint16 depth image (mm -> m)
        self.declare_parameter('min_valid_depth_m', 0.05)
        self.declare_parameter('near_mask_dilate_px', 5)
        self.declare_parameter('attach_depth_to_keypoints', True)
        self.declare_parameter('max_depth_age_sec', 0.20)

        # Future hook: dynamic workspace ROI for UR5e base exclusion
        self.declare_parameter('use_workspace_roi', False)

        self.bridge = CvBridge()
        self.latest_near_mask = None  # type: np.ndarray | None
        self.latest_depth_raw = None  # type: np.ndarray | None
        self.latest_depth_stamp_sec = -1.0
        self.last_color_msg_stamp_sec = -1.0
        self.last_depth_msg_stamp_sec = -1.0

        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.use_depth_roi = bool(self.get_parameter('use_depth_roi').value)

        self.detector_type = self.get_parameter('detector_type').value
        device = self.get_parameter('device').value
        max_kp = int(self.get_parameter('max_num_keypoints').value)
        top_k = int(self.get_parameter('top_k').value)
        xfeat_repo_dir = str(self.get_parameter('xfeat_repo_dir').value).strip()
        nfeatures = int(self.get_parameter('nfeatures').value)

        kwargs = {}
        if self.detector_type in ('superpoint', 'aliked'):
            kwargs = {'max_num_keypoints': max_kp, 'device': device}
        elif self.detector_type == 'xfeat':
            kwargs = {'top_k': top_k, 'device': device}
            if xfeat_repo_dir:
                kwargs['repo_dir'] = xfeat_repo_dir
        elif self.detector_type in ('sift', 'orb'):
            kwargs = {'nfeatures': nfeatures}

        self.det = create_detector(self.detector_type, **kwargs)
        self.get_logger().info(f'Using detector: {self.detector_type} ({self.det.__class__.__name__})')

        in_topic = self.get_parameter('input_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        kp_topic = self.get_parameter('keypoints_topic').value
        out_topic = self.get_parameter('output_topic').value
        bin_topic = self.get_parameter('binary_output_topic').value

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        realtime_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.kp_pub = self.create_publisher(Keypoints, kp_topic, realtime_qos)
        self.color_topic_uses_compressed = self._topic_uses_compressed(in_topic)
        self.depth_topic_uses_compressed = self._topic_uses_compressed(depth_topic)
        color_msg_type = CompressedImage if self.color_topic_uses_compressed else Image
        depth_msg_type = CompressedImage if self.depth_topic_uses_compressed else Image
        self.sub_color = self.create_subscription(
            color_msg_type, in_topic, self.on_color, realtime_qos
        )
        self.sub_depth = self.create_subscription(
            depth_msg_type, depth_topic, self.on_depth, realtime_qos
        )

        self.debug_pub = None
        self.binary_pub = None
        if self.debug_mode:
            self.debug_pub = self.create_publisher(Image, out_topic, pub_qos)
            self.binary_pub = self.create_publisher(Image, bin_topic, pub_qos)

        self.get_logger().info(
            f"Subscribing color: {in_topic} "
            f"({'compressed' if self.color_topic_uses_compressed else 'raw'})"
        )
        self.get_logger().info(
            f"Subscribing depth: {depth_topic} "
            f"({'compressed' if self.depth_topic_uses_compressed else 'raw'})"
        )
        self.get_logger().info(f'Publishing keypoints: {kp_topic}')
        if self.debug_mode:
            self.get_logger().info(f'Publishing debug overlay: {out_topic}')
            self.get_logger().info(f'Publishing near-mask:     {bin_topic}')
        else:
            self.get_logger().info('Debug mode disabled.')
        self.get_logger().info(
            "Depth-to-keypoints: "
            f"{'on' if bool(self.get_parameter('attach_depth_to_keypoints').value) else 'off'} "
            f"(max_age={float(self.get_parameter('max_depth_age_sec').value):.2f}s)"
        )

    @staticmethod
    def _topic_uses_compressed(topic: str) -> bool:
        return topic.endswith('/compressed') or topic.endswith('/compressedDepth')

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def _stamp_to_sec(stamp: Any) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _decode_compressed_depth_payload(msg: CompressedImage) -> np.ndarray:
        fmt = str(msg.format).lower()
        raw = bytes(msg.data)

        if len(raw) > 12:
            payload = np.frombuffer(raw[12:], dtype=np.uint8)
            depth_img = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                if fmt.startswith('32fc1'):
                    depth_quant_a, depth_quant_b = struct.unpack('<ff', raw[4:12])
                    inv_depth = depth_img.astype(np.float32)
                    depth = np.zeros(inv_depth.shape, dtype=np.float32)
                    valid = inv_depth > 0.0
                    depth[valid] = depth_quant_a / (inv_depth[valid] - depth_quant_b)
                    depth[~np.isfinite(depth)] = 0.0
                    return depth
                return depth_img

        # Fallback for plain `/compressed` depth topics that do not carry the depth header.
        depth_img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise RuntimeError('failed to decode compressed depth payload')
        return depth_img

    def _decode_depth(self, msg: Any) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            try:
                depth_img = self.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding='passthrough'
                )
                if depth_img is not None:
                    return depth_img
            except Exception:
                pass

            try:
                return self._decode_compressed_depth_payload(msg)
            except Exception as e:
                raise RuntimeError(
                    f'compressed depth decode failed for format={msg.format}'
                ) from e
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _decode_color_bgr(self, msg: Any) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            try:
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                bgr = None
            if bgr is None:
                bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError('compressed color decode returned None')
            return bgr
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _depth_raw_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        if depth_raw.dtype == np.uint16:
            scale = float(self.get_parameter('depth_scale').value)
            return depth_raw.astype(np.float32) * scale
        return depth_raw.astype(np.float32)

    def _sample_depth_at_keypoints(
        self,
        kpts_xy: np.ndarray,
        color_shape: tuple[int, int],
    ) -> tuple[np.ndarray, float]:
        n = int(kpts_xy.shape[0])
        depth_out = np.full((n,), np.nan, dtype=np.float32)
        if n <= 0:
            return depth_out, float('nan')

        if not bool(self.get_parameter('attach_depth_to_keypoints').value):
            return depth_out, float('nan')

        depth_raw = self.latest_depth_raw
        if depth_raw is None or depth_raw.ndim != 2:
            return depth_out, float('inf')

        age_sec = float('inf')
        if self.latest_depth_stamp_sec > 0.0:
            age_sec = max(0.0, self._now_sec() - self.latest_depth_stamp_sec)
        max_age_sec = max(0.0, float(self.get_parameter('max_depth_age_sec').value))
        if age_sec > max_age_sec:
            return depth_out, age_sec

        color_h, color_w = int(color_shape[0]), int(color_shape[1])
        depth_h, depth_w = int(depth_raw.shape[0]), int(depth_raw.shape[1])
        if color_h <= 0 or color_w <= 0 or depth_h <= 0 or depth_w <= 0:
            return depth_out, age_sec

        if depth_h == color_h and depth_w == color_w:
            x = np.rint(kpts_xy[:, 0]).astype(np.int32)
            y = np.rint(kpts_xy[:, 1]).astype(np.int32)
        else:
            sx = depth_w / float(color_w)
            sy = depth_h / float(color_h)
            x = np.rint(kpts_xy[:, 0] * sx).astype(np.int32)
            y = np.rint(kpts_xy[:, 1] * sy).astype(np.int32)

        inside = (x >= 0) & (x < depth_w) & (y >= 0) & (y < depth_h)
        idx = np.where(inside)[0]
        if idx.size == 0:
            return depth_out, age_sec

        sampled = depth_raw[y[idx], x[idx]]
        sampled_m = self._depth_raw_to_meters(sampled)
        min_depth = float(self.get_parameter('min_valid_depth_m').value)
        valid = np.isfinite(sampled_m) & (sampled_m > min_depth)
        if np.any(valid):
            depth_out[idx[valid]] = sampled_m[valid].astype(np.float32)
        return depth_out, age_sec

    def _make_near_mask(self, depth_raw: np.ndarray) -> np.ndarray | None:
        if depth_raw.ndim != 2:
            return None

        if depth_raw.dtype == np.uint16:
            scale = float(self.get_parameter('depth_scale').value)
            depth_m = depth_raw.astype(np.float32) * scale
        else:
            depth_m = depth_raw.astype(np.float32)

        min_depth = float(self.get_parameter('min_valid_depth_m').value)
        valid = np.isfinite(depth_m) & (depth_m > min_depth)
        if not np.any(valid):
            return None

        valid_depth = depth_m[valid]
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())

        near_mask = np.zeros(depth_m.shape, dtype=np.uint8)
        if (d_max - d_min) < 1e-6:
            near_mask[valid] = 255
        else:
            scaled = np.zeros(depth_m.shape, dtype=np.uint8)
            scaled_valid = ((valid_depth - d_min) * (255.0 / (d_max - d_min))).astype(np.uint8)
            scaled[valid] = scaled_valid

            # Otsu threshold on valid depth pixels only (near == smaller depth)
            otsu_thr, _ = cv2.threshold(
                scaled[valid].reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            near_mask[valid & (scaled <= otsu_thr)] = 255

        dilate_px = int(self.get_parameter('near_mask_dilate_px').value)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            near_mask = cv2.dilate(near_mask, kernel, iterations=1)

        return self._apply_workspace_roi(near_mask)

    def _apply_workspace_roi(self, near_mask: np.ndarray) -> np.ndarray:
        if not bool(self.get_parameter('use_workspace_roi').value):
            return near_mask

        # TODO(UR5e workspace ROI): apply a dynamically projected workspace mask
        # to suppress static robot-base regions once calibration is available.
        return near_mask

    def on_depth(self, msg: Any):
        try:
            msg_stamp_sec = self._stamp_to_sec(msg.header.stamp)
            if np.isfinite(msg_stamp_sec) and msg_stamp_sec > 0.0:
                if self.last_depth_msg_stamp_sec > 0.0 and msg_stamp_sec <= self.last_depth_msg_stamp_sec:
                    return
                self.last_depth_msg_stamp_sec = msg_stamp_sec

            use_depth_roi = bool(self.get_parameter('use_depth_roi').value)
            attach_depth = bool(self.get_parameter('attach_depth_to_keypoints').value)
            if not use_depth_roi and not attach_depth:
                self.latest_near_mask = None
                self.latest_depth_raw = None
                self.latest_depth_stamp_sec = -1.0
                return

            depth_raw = self._decode_depth(msg)
            self.latest_depth_raw = depth_raw
            stamp_sec = self._stamp_to_sec(msg.header.stamp)
            if not np.isfinite(stamp_sec) or stamp_sec <= 0.0:
                stamp_sec = self._now_sec()
            self.latest_depth_stamp_sec = stamp_sec

            if use_depth_roi:
                self.latest_near_mask = self._make_near_mask(depth_raw)
            else:
                self.latest_near_mask = None
        except Exception as e:
            self.latest_near_mask = None
            self.get_logger().warn(f'depth callback error: {e}')

    def _filter_keypoints(
        self,
        kpts_xy: np.ndarray,
        desc: np.ndarray | None,
        scores: np.ndarray | None,
        near_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
        n = int(kpts_xy.shape[0])
        keep = np.ones((n,), dtype=bool)

        if self.use_depth_roi and near_mask is None:
            keep = np.zeros((n,), dtype=bool)
        elif self.use_depth_roi and n > 0:
            h, w = near_mask.shape
            x = np.rint(kpts_xy[:, 0]).astype(np.int32)
            y = np.rint(kpts_xy[:, 1]).astype(np.int32)

            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            keep = np.zeros((n,), dtype=bool)
            idx = np.where(inside)[0]
            if idx.size > 0:
                keep[idx] = near_mask[y[idx], x[idx]] > 0

        kpts_out = kpts_xy[keep]
        desc_out = desc[keep] if desc is not None else None
        scores_out = scores[keep] if scores is not None else None
        return kpts_out, desc_out, scores_out, keep

    def on_color(self, msg: Any):
        try:
            msg_stamp_sec = self._stamp_to_sec(msg.header.stamp)
            if np.isfinite(msg_stamp_sec) and msg_stamp_sec > 0.0:
                if self.last_color_msg_stamp_sec > 0.0 and msg_stamp_sec <= self.last_color_msg_stamp_sec:
                    return
                self.last_color_msg_stamp_sec = msg_stamp_sec

            self.use_depth_roi = bool(self.get_parameter('use_depth_roi').value)
            bgr = self._decode_color_bgr(msg)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            res = self.det.detect_and_compute(gray)
            all_kpts = res.kpts_xy

            near_mask = self.latest_near_mask
            if self.use_depth_roi and near_mask is not None and near_mask.shape != gray.shape:
                near_mask = cv2.resize(near_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            pub_kpts, pub_desc, pub_scores, keep_mask = self._filter_keypoints(
                all_kpts,
                res.desc,
                res.scores,
                near_mask,
            )
            pub_depth_m, depth_age_sec = self._sample_depth_at_keypoints(pub_kpts, gray.shape)

            kp_msg = Keypoints()
            kp_msg.header = msg.header
            kp_msg.xy = pub_kpts.reshape(-1).astype(np.float32).tolist()
            if bool(self.get_parameter('attach_depth_to_keypoints').value):
                kp_msg.depth_m = pub_depth_m.astype(np.float32).reshape(-1).tolist()
            else:
                kp_msg.depth_m = []

            if pub_desc is None:
                kp_msg.descriptor_dim = 0
                kp_msg.descriptors = []
            else:
                desc = pub_desc if pub_desc.dtype == np.float32 else pub_desc.astype(np.float32)
                kp_msg.descriptor_dim = int(desc.shape[1])
                kp_msg.descriptors = desc.reshape(-1).tolist()

            if pub_scores is None:
                kp_msg.scores = []
            else:
                kp_msg.scores = pub_scores.astype(np.float32).reshape(-1).tolist()

            self.kp_pub.publish(kp_msg)

            if self.debug_mode and self.debug_pub is not None and self.binary_pub is not None:
                overlay = bgr.copy()

                # Green: all detected keypoints
                for x, y in all_kpts:
                    cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)

                # Red: keypoints that are actually published after ROI filtering
                if keep_mask.size == all_kpts.shape[0]:
                    for i, (x, y) in enumerate(all_kpts[keep_mask]):
                        if i < pub_depth_m.shape[0] and np.isfinite(pub_depth_m[i]):
                            c = (0, 0, 255)
                        else:
                            c = (255, 0, 0)
                        cv2.circle(overlay, (int(x), int(y)), 2, c, -1)

                valid_depth = int(np.count_nonzero(np.isfinite(pub_depth_m)))
                median_depth = (
                    float(np.nanmedian(pub_depth_m))
                    if valid_depth > 0 else float('nan')
                )
                depth_age_ms = depth_age_sec * 1000.0 if np.isfinite(depth_age_sec) else float('nan')
                depth_med_txt = f"{median_depth:.3f}m" if np.isfinite(median_depth) else "n/a"
                depth_age_txt = f"{depth_age_ms:.0f}ms" if np.isfinite(depth_age_ms) else "n/a"
                cv2.putText(
                    overlay,
                    f"depth valid={valid_depth}/{pub_kpts.shape[0]} med={depth_med_txt} age={depth_age_txt}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                dbg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                dbg.header = msg.header
                self.debug_pub.publish(dbg)

                if near_mask is None:
                    near_mask = np.zeros(gray.shape, dtype=np.uint8)
                mask_msg = self.bridge.cv2_to_imgmsg(near_mask, encoding='mono8')
                mask_msg.header = msg.header
                self.binary_pub.publish(mask_msg)

            self.get_logger().debug(
                f'kpts_all={all_kpts.shape[0]} kpts_pub={pub_kpts.shape[0]} '
                f'roi={"on" if self.use_depth_roi else "off"} '
                f'depth_valid={int(np.count_nonzero(np.isfinite(pub_depth_m)))}/{pub_kpts.shape[0]}'
            )
        except Exception as e:
            self.get_logger().error(f'color callback error: {e}')


def main():
    rclpy.init()
    node = KeypointNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
