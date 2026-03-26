import json
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from ibvs_msgs.msg import Keypoints, Matches


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


class IbvsTwistControllerNode(Node):
    def __init__(self):
        super().__init__('ibvs_twist_controller_node')

        # Legacy alias: historically this was the only matches topic (raw).
        self.declare_parameter('matches_topic', '/ibvs/matches')
        self.declare_parameter('raw_matches_topic', self.get_parameter('matches_topic').value)
        self.declare_parameter('filtered_matches_topic', '/ibvs/filtered_features')
        self.declare_parameter('feature_source', 'filtered')  # filtered | raw
        self.declare_parameter('filtered_fallback_to_raw', True)

        self.declare_parameter('reference_topic', '/ibvs/reference/keypoints')
        self.declare_parameter('init_done_topic', '/ibvs/init_done')
        self.declare_parameter('twist_topic', '/cartesian_twist_passthrough_controller/cmd_vel')
        self.declare_parameter('goal_reached_topic', '/ibvs/control/goal_reached')
        self.declare_parameter('publish_wx_debug', True)
        self.declare_parameter('wx_debug_topic', '/ibvs/control/wx_debug')

        # Camera intrinsics for pixel -> normalized conversion.
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # IBVS core parameters.
        self.declare_parameter('lambda_gain', 0.22)
        self.declare_parameter('dls_damping', 0.25)
        self.declare_parameter('z_est', 0.25)
        self.declare_parameter('use_per_keypoint_depth', True)
        self.declare_parameter('min_valid_depth_m', 0.05)
        self.declare_parameter('max_valid_depth_m', 3.0)
        self.declare_parameter('use_depth_fallback_ema', True)
        self.declare_parameter('depth_fallback_ema_alpha', 0.35)
        self.declare_parameter('publish_rate_hz', 30.0)

        # Safety and stop criteria.
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('require_init_done', True)
        self.declare_parameter('min_matches', 5)
        self.declare_parameter('match_timeout_sec', 0.25)
        self.declare_parameter('error_stop_px', 6.0)
        self.declare_parameter('stop_hold_sec', 1.2)
        self.declare_parameter('max_linear_speed', 0.012)
        self.declare_parameter('max_angular_speed', 0.08)
        self.declare_parameter('smooth_cmd_enable', True)
        self.declare_parameter('smooth_cmd_use_median', True)
        self.declare_parameter('smooth_cmd_median_window', 3)  # odd: 1,3,5...
        self.declare_parameter('smooth_cmd_ema_alpha', 0.35)   # 0..1, higher=faster
        self.declare_parameter('smooth_cmd_max_linear_accel', 0.08)   # m/s^2
        self.declare_parameter('smooth_cmd_max_angular_accel', 0.70)  # rad/s^2
        self.declare_parameter('log_period_sec', 1.0)

        # Allowed DOFs: default x,y,z + yaw.
        self.declare_parameter('allow_vx', True)
        self.declare_parameter('allow_vy', True)
        self.declare_parameter('allow_vz', True)
        self.declare_parameter('allow_wx', True)
        self.declare_parameter('allow_wy', True)
        self.declare_parameter('allow_wz', True)

        # Legacy compatibility parameters; no longer used for transform logic.
        self.declare_parameter('axis_sign_vx', -1.0)
        self.declare_parameter('axis_sign_vy', 1.0)
        self.declare_parameter('axis_sign_vz', -1.0)
        self.declare_parameter('axis_sign_wx', -1.0)
        self.declare_parameter('axis_sign_wy', 1.0)
        self.declare_parameter('axis_sign_wz', -1.0)
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('tcp_frame', 'tool0')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('tf_lookup_timeout_sec', 0.2)

        self.ref_xy: Optional[np.ndarray] = None

        self.last_raw_ref_id: Optional[np.ndarray] = None
        self.last_raw_cur_xy: Optional[np.ndarray] = None
        self.last_raw_depth_m: Optional[np.ndarray] = None
        self.last_raw_time_sec: float = -1.0

        self.last_filtered_ref_id: Optional[np.ndarray] = None
        self.last_filtered_cur_xy: Optional[np.ndarray] = None
        self.last_filtered_depth_m: Optional[np.ndarray] = None
        self.last_filtered_time_sec: float = -1.0

        self.init_done: bool = False
        self.goal_reached: bool = False
        self.goal_hold_start_sec: Optional[float] = None
        self.last_log_sec: float = 0.0
        self._last_change_log_text: dict[str, str] = {}
        self.depth_fallback_m: float = float(self.get_parameter('z_est').value)
        self.last_cmd_v6 = np.zeros((6,), dtype=np.float64)
        self.last_cmd_time_sec: float = -1.0
        self._median_window_cached: int = 3
        self._cmd_median_buf = [deque(maxlen=self._median_window_cached) for _ in range(6)]
        self._tf_ready_tcp_cam = False
        self._tf_ready_base_tcp = False
        self._R_tcp_cam = np.eye(3, dtype=np.float64)
        self._p_tcp_cam = np.zeros((3,), dtype=np.float64)
        self._R_base_tcp = np.eye(3, dtype=np.float64)
        self._last_tf_warn_sec = 0.0

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ref_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        init_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_ref = self.create_subscription(
            Keypoints,
            self.get_parameter('reference_topic').value,
            self.on_reference,
            ref_qos,
        )
        self.sub_raw_matches = self.create_subscription(
            Matches,
            self.get_parameter('raw_matches_topic').value,
            self.on_raw_matches,
            qos_profile_sensor_data,
        )
        self.sub_filtered_matches = self.create_subscription(
            Matches,
            self.get_parameter('filtered_matches_topic').value,
            self.on_filtered_matches,
            qos_profile_sensor_data,
        )
        self.sub_init_done = self.create_subscription(
            Bool,
            self.get_parameter('init_done_topic').value,
            self.on_init_done,
            init_qos,
        )

        self.twist_pub = self.create_publisher(
            Twist,
            self.get_parameter('twist_topic').value,
            10,
        )
        self.goal_pub = self.create_publisher(
            Bool,
            self.get_parameter('goal_reached_topic').value,
            init_qos,
        )
        self.wx_debug_pub = None
        if bool(self.get_parameter('publish_wx_debug').value):
            self.wx_debug_pub = self.create_publisher(
                String,
                self.get_parameter('wx_debug_topic').value,
                10,
            )

        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        period = 1.0 / max(rate_hz, 1e-3)
        self.timer = self.create_timer(period, self.on_timer)
        self.tf_timer = self.create_timer(0.5, self._update_tf_cache)

        self.publish_goal(False, force=True)

        self.get_logger().info(
            f"Sub raw={self.get_parameter('raw_matches_topic').value} "
            f"Sub filtered={self.get_parameter('filtered_matches_topic').value} "
            f"feature_source={self.get_parameter('feature_source').value}"
        )
        self.get_logger().info(
            "Depth fallback: "
            f"mode={'ema' if bool(self.get_parameter('use_depth_fallback_ema').value) else 'mean'} "
            f"alpha={float(self.get_parameter('depth_fallback_ema_alpha').value):.2f} "
            f"init={self.depth_fallback_m:.3f}m"
        )
        self.get_logger().info(f"Sub ref={self.get_parameter('reference_topic').value}")
        self.get_logger().info(f"Pub twist={self.get_parameter('twist_topic').value}")
        if self.wx_debug_pub is not None:
            self.get_logger().info(f"Pub wx debug={self.get_parameter('wx_debug_topic').value}")
        self.get_logger().info(
            "Cmd smoothing: "
            f"enable={bool(self.get_parameter('smooth_cmd_enable').value)} "
            f"median={bool(self.get_parameter('smooth_cmd_use_median').value)} "
            f"w={int(self.get_parameter('smooth_cmd_median_window').value)} "
            f"alpha={float(self.get_parameter('smooth_cmd_ema_alpha').value):.2f}"
        )
        self.get_logger().info(
            "TF command conversion: "
            f"camera={self._camera_frame()} tcp={self._tcp_frame()} base={self._base_frame()}"
        )

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _base_frame(self) -> str:
        return str(self.get_parameter('base_frame').value)

    def _tcp_frame(self) -> str:
        return str(self.get_parameter('tcp_frame').value)

    def _camera_frame(self) -> str:
        return str(self.get_parameter('camera_frame').value)

    def _update_tf_cache(self):
        timeout_sec = max(1e-3, float(self.get_parameter('tf_lookup_timeout_sec').value))

        try:
            tf_base_tcp = self.tf_buffer.lookup_transform(
                self._base_frame(),
                self._tcp_frame(),
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
            q_bt = tf_base_tcp.transform.rotation
            self._R_base_tcp = quat_to_rotmat(q_bt.x, q_bt.y, q_bt.z, q_bt.w)
            first_ready = not self._tf_ready_base_tcp
            self._tf_ready_base_tcp = True
            if first_ready:
                self.get_logger().info(
                    f"TF ready ({self._base_frame()} <- {self._tcp_frame()})"
                )
        except TransformException as exc:
            now = self.now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"TF lookup failed ({self._base_frame()} <- {self._tcp_frame()}): {exc}"
                )

        try:
            tf_tcp_cam = self.tf_buffer.lookup_transform(
                self._tcp_frame(),
                self._camera_frame(),
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
            t = tf_tcp_cam.transform.translation
            q = tf_tcp_cam.transform.rotation
            self._p_tcp_cam = np.array([t.x, t.y, t.z], dtype=np.float64)
            self._R_tcp_cam = quat_to_rotmat(q.x, q.y, q.z, q.w)
            first_ready = not self._tf_ready_tcp_cam
            self._tf_ready_tcp_cam = True
            if first_ready:
                p = self._p_tcp_cam
                self.get_logger().info(
                    f"TF ready ({self._tcp_frame()} <- {self._camera_frame()}), "
                    f"p_tcp_cam=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]"
                )
        except TransformException as exc:
            now = self.now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"TF lookup failed ({self._tcp_frame()} <- {self._camera_frame()}): {exc}"
                )

    def _cam_twist_to_tcp_twist(self, cam_v6: np.ndarray) -> Optional[np.ndarray]:
        if not self._tf_ready_tcp_cam:
            return None
        R = self._R_tcp_cam.copy()
        p = self._p_tcp_cam.copy()

        v_cam = cam_v6[0:3]
        w_cam = cam_v6[3:6]
        w_tcp = R @ w_cam
        v_tcp = (R @ v_cam) - np.cross(w_tcp, p)

        out = np.zeros((6,), dtype=np.float64)
        out[0:3] = v_tcp
        out[3:6] = w_tcp
        return out

    def _tcp_twist_to_base_twist(self, tcp_v6: np.ndarray) -> Optional[np.ndarray]:
        if not self._tf_ready_base_tcp:
            return None
        R = self._R_base_tcp.copy()
        out = np.zeros((6,), dtype=np.float64)
        out[0:3] = R @ tcp_v6[0:3]
        out[3:6] = R @ tcp_v6[3:6]
        return out

    def _cam_twist_to_base_twist(self, cam_v6: np.ndarray) -> Optional[np.ndarray]:
        tcp_v6 = self._cam_twist_to_tcp_twist(cam_v6)
        if tcp_v6 is None:
            return None
        return self._tcp_twist_to_base_twist(tcp_v6)

    def on_init_done(self, msg: Bool):
        self.init_done = bool(msg.data)

    def on_reference(self, msg: Keypoints):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn('Reference xy length not even; ignoring.')
            return
        self.ref_xy = xy.reshape(-1, 2)
        self.log_info_once_on_change(
            'reference_cached_for_control',
            f'Reference cached for control: K={self.ref_xy.shape[0]}',
        )

    def _cache_matches(self, msg: Matches):
        if self.ref_xy is None:
            return None, None, None

        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            return None, None, None
        cur_xy = xy.reshape(-1, 2)
        n_pairs = min(cur_xy.shape[0], len(msg.ref_id))
        if n_pairs <= 0:
            return None, None, None

        ref_id = np.asarray(msg.ref_id[:n_pairs], dtype=np.int64)
        cur_xy = cur_xy[:n_pairs]
        if len(msg.depth_m) >= n_pairs:
            depth_m = np.asarray(msg.depth_m[:n_pairs], dtype=np.float32)
        else:
            depth_m = np.full((n_pairs,), np.nan, dtype=np.float32)
        valid = (ref_id >= 0) & (ref_id < self.ref_xy.shape[0])
        if not np.any(valid):
            return None, None, None

        return ref_id[valid], cur_xy[valid], depth_m[valid]

    def on_raw_matches(self, msg: Matches):
        ref_id, cur_xy, depth_m = self._cache_matches(msg)
        if ref_id is None:
            return
        self.last_raw_ref_id = ref_id
        self.last_raw_cur_xy = cur_xy
        self.last_raw_depth_m = depth_m
        self.last_raw_time_sec = self.now_sec()

    def on_filtered_matches(self, msg: Matches):
        ref_id, cur_xy, depth_m = self._cache_matches(msg)
        if ref_id is None:
            return
        self.last_filtered_ref_id = ref_id
        self.last_filtered_cur_xy = cur_xy
        self.last_filtered_depth_m = depth_m
        self.last_filtered_time_sec = self.now_sec()

    def _active_mask(self) -> np.ndarray:
        return np.array([
            bool(self.get_parameter('allow_vx').value),
            bool(self.get_parameter('allow_vy').value),
            bool(self.get_parameter('allow_vz').value),
            bool(self.get_parameter('allow_wx').value),
            bool(self.get_parameter('allow_wy').value),
            bool(self.get_parameter('allow_wz').value),
        ], dtype=bool)

    def _pixels_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        if fx <= 1e-9 or fy <= 1e-9:
            raise ValueError('fx and fy must be > 0')

        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        return np.stack([x, y], axis=1).astype(np.float64)

    @staticmethod
    def _build_interaction_matrix(cur_n: np.ndarray, z_per_point: np.ndarray) -> np.ndarray:
        n = cur_n.shape[0]
        if z_per_point.shape[0] != n:
            raise ValueError('z_per_point length mismatch')
        L = np.zeros((2 * n, 6), dtype=np.float64)

        for i, (x, y) in enumerate(cur_n):
            z_inv = 1.0 / max(float(z_per_point[i]), 1e-6)
            r = 2 * i
            L[r, :] = [-z_inv, 0.0, x * z_inv, x * y, -(1.0 + x * x), y]
            L[r + 1, :] = [0.0, -z_inv, y * z_inv, 1.0 + y * y, -x * y, -x]

        return L

    def _update_depth_fallback(self, depth_m: Optional[np.ndarray]) -> tuple[float, int]:
        z_est = float(self.get_parameter('z_est').value)
        if (not np.isfinite(self.depth_fallback_m)) or self.depth_fallback_m <= 0.0:
            self.depth_fallback_m = z_est

        if depth_m is None or depth_m.size == 0:
            return self.depth_fallback_m, 0

        min_depth = float(self.get_parameter('min_valid_depth_m').value)
        max_depth = float(self.get_parameter('max_valid_depth_m').value)
        d = np.asarray(depth_m, dtype=np.float64)
        valid = np.isfinite(d) & (d > min_depth)
        if max_depth > min_depth:
            valid = valid & (d < max_depth)
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            return self.depth_fallback_m, 0

        frame_mean = float(np.mean(d[valid]))
        if bool(self.get_parameter('use_depth_fallback_ema').value):
            alpha = float(self.get_parameter('depth_fallback_ema_alpha').value)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            self.depth_fallback_m = alpha * frame_mean + (1.0 - alpha) * self.depth_fallback_m
        else:
            self.depth_fallback_m = frame_mean

        if (not np.isfinite(self.depth_fallback_m)) or self.depth_fallback_m <= 0.0:
            self.depth_fallback_m = z_est
        return self.depth_fallback_m, valid_count

    def _build_depth_vector(
        self,
        depth_m: Optional[np.ndarray],
        n: int,
        fallback_z: float,
    ) -> tuple[np.ndarray, int]:
        out = np.full((n,), fallback_z, dtype=np.float64)
        if n <= 0:
            return out, 0

        use_depth = bool(self.get_parameter('use_per_keypoint_depth').value)
        if (not use_depth) or depth_m is None or depth_m.shape[0] != n:
            return out, 0

        min_depth = float(self.get_parameter('min_valid_depth_m').value)
        max_depth = float(self.get_parameter('max_valid_depth_m').value)
        d = np.asarray(depth_m, dtype=np.float64)
        valid = np.isfinite(d) & (d > min_depth)
        if max_depth > min_depth:
            valid = valid & (d < max_depth)
        if np.any(valid):
            out[valid] = d[valid]
        return out, int(np.count_nonzero(valid))

    def _compute_wx_diagnostics(
        self,
        cur_xy: np.ndarray,
        err_vec: np.ndarray,
        L: np.ndarray,
        damp: float,
        gain: float,
    ) -> dict:
        n = int(cur_xy.shape[0])
        cy_px = float(self.get_parameter('cy').value)
        out = {
            'wx_num': None,
            'wx_den': None,
            'wx_scalar_pre': None,
            'wx_num_top': None,
            'wx_num_bottom': None,
            'wx_abs_top': None,
            'wx_abs_bottom': None,
            'n_top': 0,
            'n_bottom': 0,
            'y_split_px': cy_px,
        }
        if n <= 0 or L.shape[0] != (2 * n) or L.shape[1] < 4 or err_vec.size != (2 * n):
            return out

        wx_col = L[:, 3]
        num = float(np.dot(wx_col, err_vec))
        den = float(np.dot(wx_col, wx_col) + damp * damp)
        out['wx_num'] = num
        out['wx_den'] = den
        if den > 1e-12 and np.isfinite(den):
            out['wx_scalar_pre'] = float(-gain * (num / den))

        contrib = (wx_col.reshape(-1, 2) * err_vec.reshape(-1, 2)).sum(axis=1)
        top_mask = cur_xy[:, 1] <= cy_px
        bottom_mask = ~top_mask
        out['n_top'] = int(np.count_nonzero(top_mask))
        out['n_bottom'] = int(np.count_nonzero(bottom_mask))

        if out['n_top'] > 0:
            c_top = contrib[top_mask]
            out['wx_num_top'] = float(np.sum(c_top))
            out['wx_abs_top'] = float(np.sum(np.abs(c_top)))
        if out['n_bottom'] > 0:
            c_bottom = contrib[bottom_mask]
            out['wx_num_bottom'] = float(np.sum(c_bottom))
            out['wx_abs_bottom'] = float(np.sum(np.abs(c_bottom)))
        return out

    def _compute_ibvs_twist(
        self,
        cur_xy: np.ndarray,
        des_xy: np.ndarray,
        depth_m: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int, int, float, float, dict]:
        cur_n = self._pixels_to_normalized(cur_xy)
        des_n = self._pixels_to_normalized(des_xy)
        err_vec = (cur_n - des_n).reshape(-1)

        fallback_z, depth_for_fallback = self._update_depth_fallback(depth_m)
        z_vec, depth_valid = self._build_depth_vector(depth_m, cur_n.shape[0], fallback_z)
        L = self._build_interaction_matrix(cur_n, z_vec)
        damp = float(self.get_parameter('dls_damping').value)
        gain = float(self.get_parameter('lambda_gain').value)
        wx_dbg = self._compute_wx_diagnostics(cur_xy, err_vec, L, damp, gain)

        active = self._active_mask()
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            wx_dbg['wx_cmd_pre'] = 0.0
            return (
                np.zeros((6,), dtype=np.float64),
                depth_valid,
                depth_for_fallback,
                float(np.median(z_vec)),
                fallback_z,
                wx_dbg,
            )

        L_a = L[:, active_idx]

        I = np.eye(active_idx.size, dtype=np.float64)
        lhs = L_a.T @ L_a + (damp * damp) * I
        rhs = L_a.T @ err_vec

        try:
            v_a = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            v_a = np.linalg.pinv(L_a) @ err_vec

        v6 = np.zeros((6,), dtype=np.float64)
        v6[active_idx] = -gain * v_a
        wx_dbg['wx_cmd_pre'] = float(v6[3])
        return v6, depth_valid, depth_for_fallback, float(np.median(z_vec)), fallback_z, wx_dbg

    def _publish_wx_debug(
        self,
        now: float,
        source: str,
        points: int,
        rms_px: float,
        wx_dbg: dict,
        depth_valid: int,
        depth_for_fallback: int,
        depth_median: float,
        depth_fallback: float,
        v6_post: np.ndarray,
    ):
        if self.wx_debug_pub is None:
            return
        payload = {
            't': now,
            'source': source,
            'points': int(points),
            'rms_px': float(rms_px),
            'allow_wx': bool(self.get_parameter('allow_wx').value),
            'base_frame': self._base_frame(),
            'tcp_frame': self._tcp_frame(),
            'camera_frame': self._camera_frame(),
            'wx_cmd_pre': wx_dbg.get('wx_cmd_pre', None),
            'wx_cmd_post': float(v6_post[3]),
            'wx_num': wx_dbg.get('wx_num', None),
            'wx_den': wx_dbg.get('wx_den', None),
            'wx_scalar_pre': wx_dbg.get('wx_scalar_pre', None),
            'wx_num_top': wx_dbg.get('wx_num_top', None),
            'wx_num_bottom': wx_dbg.get('wx_num_bottom', None),
            'wx_abs_top': wx_dbg.get('wx_abs_top', None),
            'wx_abs_bottom': wx_dbg.get('wx_abs_bottom', None),
            'n_top': int(wx_dbg.get('n_top', 0)),
            'n_bottom': int(wx_dbg.get('n_bottom', 0)),
            'y_split_px': wx_dbg.get('y_split_px', None),
            'depth_valid': int(depth_valid),
            'depth_samples': int(depth_for_fallback),
            'depth_median_m': float(depth_median) if np.isfinite(depth_median) else None,
            'depth_fallback_m': float(depth_fallback) if np.isfinite(depth_fallback) else None,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(',', ':'))
        self.wx_debug_pub.publish(msg)

    def _apply_active_mask(self, v6: np.ndarray) -> np.ndarray:
        out = v6.copy()
        active = self._active_mask()
        out[~active] = 0.0
        return out

    def _apply_speed_limits(self, v6: np.ndarray) -> np.ndarray:
        out = v6.copy()
        max_lin = abs(float(self.get_parameter('max_linear_speed').value))
        max_ang = abs(float(self.get_parameter('max_angular_speed').value))
        out[0:3] = np.clip(out[0:3], -max_lin, max_lin)
        out[3:6] = np.clip(out[3:6], -max_ang, max_ang)
        return out

    @staticmethod
    def _as_odd_window(n: int) -> int:
        w = max(1, int(n))
        if (w % 2) == 0:
            w += 1
        return w

    def _apply_cmd_smoothing(self, v6: np.ndarray, now: float) -> np.ndarray:
        out = np.asarray(v6, dtype=np.float64).copy()
        if not bool(self.get_parameter('smooth_cmd_enable').value):
            self.last_cmd_v6 = out.copy()
            self.last_cmd_time_sec = now
            return out

        use_median = bool(self.get_parameter('smooth_cmd_use_median').value)
        win = self._as_odd_window(int(self.get_parameter('smooth_cmd_median_window').value))
        if win != self._median_window_cached:
            self._median_window_cached = win
            self._cmd_median_buf = [deque(maxlen=win) for _ in range(6)]

        if use_median and win > 1:
            med = out.copy()
            for i in range(6):
                self._cmd_median_buf[i].append(float(out[i]))
                med[i] = float(np.median(np.asarray(self._cmd_median_buf[i], dtype=np.float64)))
            out = med

        alpha = float(self.get_parameter('smooth_cmd_ema_alpha').value)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.last_cmd_time_sec <= 0.0:
            ema = out
        else:
            ema = alpha * out + (1.0 - alpha) * self.last_cmd_v6

        # Slew-rate limiter to suppress single-frame command peaks.
        if self.last_cmd_time_sec > 0.0:
            dt = max(1e-3, now - self.last_cmd_time_sec)
            max_lin_acc = abs(float(self.get_parameter('smooth_cmd_max_linear_accel').value))
            max_ang_acc = abs(float(self.get_parameter('smooth_cmd_max_angular_accel').value))
            max_lin_step = max_lin_acc * dt
            max_ang_step = max_ang_acc * dt

            delta = ema - self.last_cmd_v6
            delta[0:3] = np.clip(delta[0:3], -max_lin_step, max_lin_step)
            delta[3:6] = np.clip(delta[3:6], -max_ang_step, max_ang_step)
            out = self.last_cmd_v6 + delta
        else:
            out = ema

        self.last_cmd_v6 = out.copy()
        self.last_cmd_time_sec = now
        return out

    @staticmethod
    def _to_twist(v6: np.ndarray) -> Twist:
        msg = Twist()
        msg.linear.x = float(v6[0])
        msg.linear.y = float(v6[1])
        msg.linear.z = float(v6[2])
        msg.angular.x = float(v6[3])
        msg.angular.y = float(v6[4])
        msg.angular.z = float(v6[5])
        return msg

    def publish_zero_twist(self):
        self.twist_pub.publish(Twist())
        self.last_cmd_v6 = np.zeros((6,), dtype=np.float64)
        self.last_cmd_time_sec = -1.0
        self._cmd_median_buf = [deque(maxlen=self._median_window_cached) for _ in range(6)]

    def publish_goal(self, reached: bool, force: bool = False):
        if (not force) and (reached == self.goal_reached):
            return
        self.goal_reached = reached
        msg = Bool()
        msg.data = reached
        self.goal_pub.publish(msg)

    def maybe_log_status(self, text: str):
        now = self.now_sec()
        period = max(0.1, float(self.get_parameter('log_period_sec').value))
        if (now - self.last_log_sec) < period:
            return
        self.last_log_sec = now
        self.get_logger().info(text)

    def log_info_once_on_change(self, key: str, text: str):
        if self._last_change_log_text.get(key) == text:
            return
        self._last_change_log_text[key] = text
        self.get_logger().info(text)

    def reset_change_log(self, key: str):
        self._last_change_log_text.pop(key, None)

    def _select_feature_set(self, now: float, timeout_sec: float):
        source_pref = str(self.get_parameter('feature_source').value).strip().lower()
        if source_pref not in ('filtered', 'raw'):
            source_pref = 'filtered'

        fallback = bool(self.get_parameter('filtered_fallback_to_raw').value)

        filtered_ready = (
            self.last_filtered_ref_id is not None
            and self.last_filtered_cur_xy is not None
            and self.last_filtered_depth_m is not None
            and self.last_filtered_time_sec > 0.0
            and (now - self.last_filtered_time_sec) <= timeout_sec
        )
        raw_ready = (
            self.last_raw_ref_id is not None
            and self.last_raw_cur_xy is not None
            and self.last_raw_depth_m is not None
            and self.last_raw_time_sec > 0.0
            and (now - self.last_raw_time_sec) <= timeout_sec
        )

        if source_pref == 'filtered':
            if filtered_ready:
                return (
                    'filtered',
                    self.last_filtered_ref_id,
                    self.last_filtered_cur_xy,
                    self.last_filtered_depth_m,
                )
            if fallback and raw_ready:
                return (
                    'raw(fallback)',
                    self.last_raw_ref_id,
                    self.last_raw_cur_xy,
                    self.last_raw_depth_m,
                )
            return None, None, None, None

        if raw_ready:
            return 'raw', self.last_raw_ref_id, self.last_raw_cur_xy, self.last_raw_depth_m
        return None, None, None, None

    def on_timer(self):
        now = self.now_sec()

        enable_motion = bool(self.get_parameter('enable_motion').value)
        require_init_done = bool(self.get_parameter('require_init_done').value)
        if enable_motion:
            self.reset_change_log('motion_disabled_enable_motion_false')
        if (not enable_motion) or (require_init_done and not self.init_done):
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            if not enable_motion:
                self.log_info_once_on_change(
                    'motion_disabled_enable_motion_false',
                    'Motion disabled (enable_motion=false).',
                )
            else:
                self.maybe_log_status('Waiting for /ibvs/init_done=true before moving.')
            return

        if self.ref_xy is None:
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            self.maybe_log_status('No reference keypoints cached yet.')
            return

        timeout_sec = float(self.get_parameter('match_timeout_sec').value)
        source, ref_id, cur_xy, depth_m = self._select_feature_set(now, timeout_sec)
        if source is None:
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            self.maybe_log_status('No fresh feature set available (selected source stale/unavailable).')
            return

        des_xy = self.ref_xy[ref_id]
        if cur_xy.shape[0] == 0:
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            self.maybe_log_status('Selected feature set is empty.')
            return

        min_matches = int(self.get_parameter('min_matches').value)
        if cur_xy.shape[0] < min_matches:
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            self.maybe_log_status(
                f'Not enough features: {cur_xy.shape[0]} < min_matches={min_matches}.'
            )
            return

        err_px = cur_xy - des_xy
        rms_px = float(np.sqrt(np.mean(np.sum(err_px * err_px, axis=1))))

        stop_thr_px = float(self.get_parameter('error_stop_px').value)
        stop_hold_sec = float(self.get_parameter('stop_hold_sec').value)
        if rms_px <= stop_thr_px:
            if self.goal_hold_start_sec is None:
                self.goal_hold_start_sec = now
            if (now - self.goal_hold_start_sec) >= stop_hold_sec:
                self.publish_goal(True)
                self.publish_zero_twist()
                self.maybe_log_status(
                    f'Goal reached: source={source} rms_px={rms_px:.2f}, points={cur_xy.shape[0]}.'
                )
                return
        else:
            self.goal_hold_start_sec = None
            self.publish_goal(False)

        try:
            v6, depth_valid, depth_for_fallback, depth_median, depth_fallback, wx_dbg = self._compute_ibvs_twist(
                cur_xy, des_xy, depth_m
            )
            v6_pre = v6.copy()
            v6 = self._apply_active_mask(v6)
            v6 = self._apply_speed_limits(v6)
            v6 = self._apply_cmd_smoothing(v6, now)
            v6 = self._apply_speed_limits(v6)
            wx_dbg['wx_cmd_pre'] = float(v6_pre[3])
            v6_base = self._cam_twist_to_base_twist(v6)
            if v6_base is None:
                raise RuntimeError('TF transform camera->tcp->base unavailable')
            v6_base = self._apply_speed_limits(v6_base)
            if not np.all(np.isfinite(v6_base)):
                raise ValueError('non-finite twist computed')
        except Exception as exc:
            self.publish_zero_twist()
            self.maybe_log_status(f'IBVS solve failed ({exc}), command set to zero.')
            return

        self._publish_wx_debug(
            now,
            source,
            cur_xy.shape[0],
            rms_px,
            wx_dbg,
            depth_valid,
            depth_for_fallback,
            depth_median,
            depth_fallback,
            v6_base,
        )
        self.twist_pub.publish(self._to_twist(v6_base))
        depth_median_txt = f'{depth_median:.3f}' if np.isfinite(depth_median) else 'n/a'
        depth_fallback_txt = f'{depth_fallback:.3f}' if np.isfinite(depth_fallback) else 'n/a'
        self.maybe_log_status(
            f"IBVS active: source={source} points={cur_xy.shape[0]} rms_px={rms_px:.2f} "
            f"depth_valid={depth_valid}/{cur_xy.shape[0]} "
            f"depth_samples={depth_for_fallback}/{cur_xy.shape[0]} "
            f"depth_med={depth_median_txt}m depth_fb={depth_fallback_txt}m "
            f"twist_base=[{v6_base[0]:+.3f},{v6_base[1]:+.3f},{v6_base[2]:+.3f},"
            f"{v6_base[3]:+.3f},{v6_base[4]:+.3f},{v6_base[5]:+.3f}]"
        )


def main():
    rclpy.init()
    node = IbvsTwistControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_zero_twist()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
