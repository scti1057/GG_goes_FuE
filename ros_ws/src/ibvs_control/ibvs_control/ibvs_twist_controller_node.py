from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Bool

from ibvs_msgs.msg import Keypoints, Matches


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

        # Camera intrinsics for pixel -> normalized conversion.
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # IBVS core parameters.
        self.declare_parameter('lambda_gain', 0.36)
        self.declare_parameter('dls_damping', 0.1)
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
        self.declare_parameter('error_stop_px', 10.0)
        self.declare_parameter('stop_hold_sec', 0.8)
        self.declare_parameter('max_linear_speed', 0.012)
        self.declare_parameter('max_angular_speed', 0.15)
        self.declare_parameter('log_period_sec', 1.0)

        # Allowed DOFs: default x,y,z + yaw.
        self.declare_parameter('allow_vx', True)
        self.declare_parameter('allow_vy', True)
        self.declare_parameter('allow_vz', True)
        self.declare_parameter('allow_wx', False)
        self.declare_parameter('allow_wy', False)
        self.declare_parameter('allow_wz', True)

        # Axis sign tuning for camera-to-tcp frame convention.
        self.declare_parameter('axis_sign_vx', -1.0)
        self.declare_parameter('axis_sign_vy', 1.0)
        self.declare_parameter('axis_sign_vz', -1.0)
        self.declare_parameter('axis_sign_wx', 1.0)
        self.declare_parameter('axis_sign_wy', 1.0)
        self.declare_parameter('axis_sign_wz', -1.0)

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

        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        period = 1.0 / max(rate_hz, 1e-3)
        self.timer = self.create_timer(period, self.on_timer)

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

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

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

    def _axis_signs(self) -> np.ndarray:
        return np.array([
            float(self.get_parameter('axis_sign_vx').value),
            float(self.get_parameter('axis_sign_vy').value),
            float(self.get_parameter('axis_sign_vz').value),
            float(self.get_parameter('axis_sign_wx').value),
            float(self.get_parameter('axis_sign_wy').value),
            float(self.get_parameter('axis_sign_wz').value),
        ], dtype=np.float64)

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

    def _compute_ibvs_twist(
        self,
        cur_xy: np.ndarray,
        des_xy: np.ndarray,
        depth_m: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int, int, float, float]:
        cur_n = self._pixels_to_normalized(cur_xy)
        des_n = self._pixels_to_normalized(des_xy)
        err_vec = (cur_n - des_n).reshape(-1)

        fallback_z, depth_for_fallback = self._update_depth_fallback(depth_m)
        z_vec, depth_valid = self._build_depth_vector(depth_m, cur_n.shape[0], fallback_z)
        L = self._build_interaction_matrix(cur_n, z_vec)

        active = self._active_mask()
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            return (
                np.zeros((6,), dtype=np.float64),
                depth_valid,
                depth_for_fallback,
                float(np.median(z_vec)),
                fallback_z,
            )

        L_a = L[:, active_idx]
        damp = float(self.get_parameter('dls_damping').value)
        gain = float(self.get_parameter('lambda_gain').value)

        I = np.eye(active_idx.size, dtype=np.float64)
        lhs = L_a.T @ L_a + (damp * damp) * I
        rhs = L_a.T @ err_vec

        try:
            v_a = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            v_a = np.linalg.pinv(L_a) @ err_vec

        v6 = np.zeros((6,), dtype=np.float64)
        v6[active_idx] = -gain * v_a
        return v6, depth_valid, depth_for_fallback, float(np.median(z_vec)), fallback_z

    def _apply_limits(self, v6: np.ndarray) -> np.ndarray:
        out = v6.copy()
        out *= self._axis_signs()

        max_lin = abs(float(self.get_parameter('max_linear_speed').value))
        max_ang = abs(float(self.get_parameter('max_angular_speed').value))
        out[0:3] = np.clip(out[0:3], -max_lin, max_lin)
        out[3:6] = np.clip(out[3:6], -max_ang, max_ang)

        active = self._active_mask()
        out[~active] = 0.0
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
            v6, depth_valid, depth_for_fallback, depth_median, depth_fallback = self._compute_ibvs_twist(
                cur_xy, des_xy, depth_m
            )
            v6 = self._apply_limits(v6)
            if not np.all(np.isfinite(v6)):
                raise ValueError('non-finite twist computed')
        except Exception as exc:
            self.publish_zero_twist()
            self.maybe_log_status(f'IBVS solve failed ({exc}), command set to zero.')
            return

        self.twist_pub.publish(self._to_twist(v6))
        depth_median_txt = f'{depth_median:.3f}' if np.isfinite(depth_median) else 'n/a'
        depth_fallback_txt = f'{depth_fallback:.3f}' if np.isfinite(depth_fallback) else 'n/a'
        self.maybe_log_status(
            f"IBVS active: source={source} points={cur_xy.shape[0]} rms_px={rms_px:.2f} "
            f"depth_valid={depth_valid}/{cur_xy.shape[0]} "
            f"depth_samples={depth_for_fallback}/{cur_xy.shape[0]} "
            f"depth_med={depth_median_txt}m depth_fb={depth_fallback_txt}m "
            f"twist=[{v6[0]:+.3f},{v6[1]:+.3f},{v6[2]:+.3f},{v6[3]:+.3f},{v6[4]:+.3f},{v6[5]:+.3f}]"
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
