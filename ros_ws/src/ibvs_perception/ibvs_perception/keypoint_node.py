import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from ibvs_msgs.msg import Keypoints
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image

from ibvs_perception.detectors import create_detector


class KeypointNode(Node):
    def __init__(self):
        super().__init__('keypoint_node')

        # Topics
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
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

        # Future hook: dynamic workspace ROI for UR5e base exclusion
        self.declare_parameter('use_workspace_roi', False)

        self.bridge = CvBridge()
        self.latest_near_mask = None  # type: np.ndarray | None

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

        self.kp_pub = self.create_publisher(Keypoints, kp_topic, 10)
        self.sub_color = self.create_subscription(Image, in_topic, self.on_color, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)

        self.debug_pub = None
        self.binary_pub = None
        if self.debug_mode:
            self.debug_pub = self.create_publisher(Image, out_topic, pub_qos)
            self.binary_pub = self.create_publisher(Image, bin_topic, pub_qos)

        self.get_logger().info(f'Subscribing color: {in_topic}')
        self.get_logger().info(f'Subscribing depth: {depth_topic}')
        self.get_logger().info(f'Publishing keypoints: {kp_topic}')
        if self.debug_mode:
            self.get_logger().info(f'Publishing debug overlay: {out_topic}')
            self.get_logger().info(f'Publishing near-mask:     {bin_topic}')
        else:
            self.get_logger().info('Debug mode disabled.')

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

    def on_depth(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_near_mask = self._make_near_mask(depth_raw)
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

    def on_color(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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

            kp_msg = Keypoints()
            kp_msg.header = msg.header
            kp_msg.xy = pub_kpts.reshape(-1).astype(np.float32).tolist()

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
                    for x, y in all_kpts[keep_mask]:
                        cv2.circle(overlay, (int(x), int(y)), 2, (0, 0, 255), -1)

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
                f'roi={"on" if self.use_depth_roi else "off"}'
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
