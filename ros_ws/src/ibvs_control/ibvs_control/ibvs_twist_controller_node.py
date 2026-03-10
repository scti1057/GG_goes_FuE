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

from ibvs_msgs.msg import Keypoints, Matches, ProxyCorners


class IbvsTwistControllerNode(Node):
    def __init__(self):
        super().__init__('ibvs_twist_controller_node')

        self.declare_parameter('matches_topic', '/ibvs/matches')
        self.declare_parameter('reference_topic', '/ibvs/reference/keypoints')
        self.declare_parameter('init_done_topic', '/ibvs/init_done')
        self.declare_parameter('twist_topic', '/cartesian_twist_passthrough_controller/cmd_vel')
        self.declare_parameter('goal_reached_topic', '/ibvs/control/goal_reached')
        self.declare_parameter('proxy_corners_topic', '/ibvs/filter/proxy_corners')

        # Camera intrinsics for pixel -> normalized conversion.
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        # IBVS core parameters.
        self.declare_parameter('lambda_gain', 0.12)
        self.declare_parameter('dls_damping', 0.1)
        self.declare_parameter('z_est', 0.25)
        self.declare_parameter('publish_rate_hz', 30.0)

        # Safety and stop criteria.
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('require_init_done', True)
        self.declare_parameter('min_matches', 60)
        self.declare_parameter('match_timeout_sec', 0.25)
        self.declare_parameter('proxy_timeout_sec', 0.25)
        self.declare_parameter('error_stop_px', 10.0)
        self.declare_parameter('stop_hold_sec', 0.8)
        self.declare_parameter('max_linear_speed', 0.004)
        self.declare_parameter('max_angular_speed', 0.05)
        self.declare_parameter('log_period_sec', 1.0)
        self.declare_parameter('use_proxy_corners', True)
        self.declare_parameter('proxy_fallback_to_matches', True)

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
        self.last_ref_id: Optional[np.ndarray] = None
        self.last_cur_xy: Optional[np.ndarray] = None
        self.last_matches_time_sec: float = -1.0
        self.last_proxy_ref_xy: Optional[np.ndarray] = None
        self.last_proxy_cur_xy: Optional[np.ndarray] = None
        self.last_proxy_time_sec: float = -1.0
        self.init_done: bool = False
        self.goal_reached: bool = False
        self.goal_hold_start_sec: Optional[float] = None
        self.last_log_sec: float = 0.0

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
        self.sub_matches = self.create_subscription(
            Matches,
            self.get_parameter('matches_topic').value,
            self.on_matches,
            qos_profile_sensor_data,
        )
        self.sub_init_done = self.create_subscription(
            Bool,
            self.get_parameter('init_done_topic').value,
            self.on_init_done,
            init_qos,
        )
        self.sub_proxy = self.create_subscription(
            ProxyCorners,
            self.get_parameter('proxy_corners_topic').value,
            self.on_proxy_corners,
            qos_profile_sensor_data,
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
            f"Sub matches={self.get_parameter('matches_topic').value} "
            f"Sub ref={self.get_parameter('reference_topic').value}"
        )
        self.get_logger().info(
            f"Sub proxy={self.get_parameter('proxy_corners_topic').value} "
            f"use_proxy_corners={self.get_parameter('use_proxy_corners').value}"
        )
        self.get_logger().info(f"Pub twist={self.get_parameter('twist_topic').value}")
        self.get_logger().info(
            "DOF mask: "
            f"vx={self.get_parameter('allow_vx').value} "
            f"vy={self.get_parameter('allow_vy').value} "
            f"vz={self.get_parameter('allow_vz').value} "
            f"wx={self.get_parameter('allow_wx').value} "
            f"wy={self.get_parameter('allow_wy').value} "
            f"wz={self.get_parameter('allow_wz').value}"
        )

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
        self.get_logger().info(f"Reference cached for control: K={self.ref_xy.shape[0]}")

    def on_matches(self, msg: Matches):
        if self.ref_xy is None:
            return

        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn('Matches xy length not even; skipping frame.')
            return
        cur_xy = xy.reshape(-1, 2)
        n_pairs = min(cur_xy.shape[0], len(msg.ref_id))
        if n_pairs <= 0:
            return

        ref_id = np.asarray(msg.ref_id[:n_pairs], dtype=np.int64)
        cur_xy = cur_xy[:n_pairs]

        valid = (ref_id >= 0) & (ref_id < self.ref_xy.shape[0])
        if not np.any(valid):
            return

        self.last_ref_id = ref_id[valid]
        self.last_cur_xy = cur_xy[valid]
        self.last_matches_time_sec = self.now_sec()

    def on_proxy_corners(self, msg: ProxyCorners):
        if not bool(msg.valid):
            return

        ref_xy = np.asarray(msg.ref_xy, dtype=np.float32)
        cur_xy = np.asarray(msg.cur_xy, dtype=np.float32)
        if ref_xy.size != 8 or cur_xy.size != 8:
            self.get_logger().warn('Proxy corners must contain exactly 4 points (8 floats).')
            return

        self.last_proxy_ref_xy = ref_xy.reshape(4, 2)
        self.last_proxy_cur_xy = cur_xy.reshape(4, 2)
        self.last_proxy_time_sec = self.now_sec()

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
    def _build_interaction_matrix(cur_n: np.ndarray, z_est: float) -> np.ndarray:
        n = cur_n.shape[0]
        L = np.zeros((2 * n, 6), dtype=np.float64)
        z_inv = 1.0 / max(z_est, 1e-6)

        for i, (x, y) in enumerate(cur_n):
            r = 2 * i
            L[r, :] = [-z_inv, 0.0, x * z_inv, x * y, -(1.0 + x * x), y]
            L[r + 1, :] = [0.0, -z_inv, y * z_inv, 1.0 + y * y, -x * y, -x]

        return L

    def _compute_ibvs_twist(self, cur_xy: np.ndarray, des_xy: np.ndarray) -> np.ndarray:
        cur_n = self._pixels_to_normalized(cur_xy)
        des_n = self._pixels_to_normalized(des_xy)
        err_vec = (cur_n - des_n).reshape(-1)

        z_est = float(self.get_parameter('z_est').value)
        L = self._build_interaction_matrix(cur_n, z_est)

        active = self._active_mask()
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            return np.zeros((6,), dtype=np.float64)

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
        return v6

    def _apply_limits(self, v6: np.ndarray) -> np.ndarray:
        out = v6.copy()
        out *= self._axis_signs()

        max_lin = abs(float(self.get_parameter('max_linear_speed').value))
        max_ang = abs(float(self.get_parameter('max_angular_speed').value))
        out[0:3] = np.clip(out[0:3], -max_lin, max_lin)
        out[3:6] = np.clip(out[3:6], -max_ang, max_ang)

        # Hard enforce disabled DOFs, even if signs/params were changed at runtime.
        active = self._active_mask()
        out[~active] = 0.0
        return out

    @staticmethod
    def _to_twist(v6: np.ndarray) -> Twist:
        msg = Twist()
        msg.linear.x = float(-v6[1])
        msg.linear.y = float(v6[0])
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

    def on_timer(self):
        now = self.now_sec()

        enable_motion = bool(self.get_parameter('enable_motion').value)
        require_init_done = bool(self.get_parameter('require_init_done').value)
        if (not enable_motion) or (require_init_done and not self.init_done):
            self.goal_hold_start_sec = None
            self.publish_goal(False)
            self.publish_zero_twist()
            if not enable_motion:
                self.maybe_log_status('Motion disabled (enable_motion=false).')
            else:
                self.maybe_log_status('Waiting for /ibvs/init_done=true before moving.')
            return

        cur_xy = None
        des_xy = None
        source = 'matches'

        use_proxy = bool(self.get_parameter('use_proxy_corners').value)
        fallback_to_matches = bool(self.get_parameter('proxy_fallback_to_matches').value)
        if use_proxy:
            proxy_timeout_sec = float(self.get_parameter('proxy_timeout_sec').value)
            proxy_ready = self.last_proxy_ref_xy is not None and self.last_proxy_cur_xy is not None
            proxy_fresh = self.last_proxy_time_sec > 0.0 and (now - self.last_proxy_time_sec) <= proxy_timeout_sec
            if proxy_ready and proxy_fresh:
                cur_xy = self.last_proxy_cur_xy
                des_xy = self.last_proxy_ref_xy
                source = 'proxy'
                self.maybe_log_status('Proxy mode active.')
            elif not fallback_to_matches:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('Proxy mode active, but proxy corners unavailable or stale.')
                return
            else:
                self.maybe_log_status('Proxy unavailable/stale, fallback to matches.')

        if cur_xy is None:
            if self.ref_xy is None:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('No reference keypoints cached yet.')
                return

            if self.last_ref_id is None or self.last_cur_xy is None:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('No matches received yet.')
                return

            timeout_sec = float(self.get_parameter('match_timeout_sec').value)
            if self.last_matches_time_sec <= 0.0 or (now - self.last_matches_time_sec) > timeout_sec:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('Match timeout, command set to zero.')
                return

            cur_xy = self.last_cur_xy
            des_xy = self.ref_xy[self.last_ref_id]
            if cur_xy.shape[0] == 0:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('Matches empty after filtering.')
                return

            min_matches = int(self.get_parameter('min_matches').value)
            if cur_xy.shape[0] < min_matches:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status(
                    f'Not enough matches: {cur_xy.shape[0]} < min_matches={min_matches}.'
                )
                return
        else:
            if cur_xy.shape[0] != 4 or des_xy.shape[0] != 4:
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('Proxy mode requires exactly 4 corners.')
                return

            if not (np.isfinite(cur_xy).all() and np.isfinite(des_xy).all()):
                self.goal_hold_start_sec = None
                self.publish_goal(False)
                self.publish_zero_twist()
                self.maybe_log_status('Proxy corners contain non-finite values.')
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
            v6 = self._compute_ibvs_twist(cur_xy, des_xy)
            v6 = self._apply_limits(v6)
            if not np.all(np.isfinite(v6)):
                raise ValueError('non-finite twist computed')
        except Exception as exc:
            self.publish_zero_twist()
            self.maybe_log_status(f'IBVS solve failed ({exc}), command set to zero.')
            return

        self.twist_pub.publish(self._to_twist(v6))
        self.maybe_log_status(
            f"IBVS active: source={source} points={cur_xy.shape[0]} rms_px={rms_px:.2f} "
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
