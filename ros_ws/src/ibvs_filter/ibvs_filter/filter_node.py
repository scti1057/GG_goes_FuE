#!/usr/bin/env python3

import threading

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from ibvs_msgs.msg import Keypoints, Matches
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32, String, UInt32

from ibvs_filter.core.ekf import ExtendedKalmanFilter
from ibvs_filter.core.eskf import ErrorStateKalmanFilter
from ibvs_filter.core.skf import StandardKalmanFilter
from ibvs_filter.core.ukf import UnscentedKalmanFilter


class FilterNode(Node):
    def __init__(self):
        super().__init__('ibvs_filter_node')

        self.declare_parameter('filter_type', 'ekf')
        self.declare_parameter('q_noise', 2.0)
        self.declare_parameter('r_noise', 1.1)
        self.declare_parameter('z_depth', 0.25)
        self.declare_parameter('gate_threshold', 20.0)
        self.declare_parameter('predict_rate', 120.0)

        self.declare_parameter('max_active_keypoints', 20)
        self.declare_parameter('min_init_keypoints', 8)
        self.declare_parameter('min_update_keypoints', 4)
        self.declare_parameter('force_relocalization', False)

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_frame')
        self.declare_parameter(
            'camera_velocity_topic',
            '/cartesian_twist_passthrough_controller/cmd_vel',
        )
        self.declare_parameter('camera_velocity_deadband_linear', 0.0)
        self.declare_parameter('camera_velocity_deadband_angular', 0.0)
        self.declare_parameter('camera_velocity_stale_timeout', 0.2)

        self.declare_parameter('filter_status_topic', '/ibvs/filter/status')
        self.declare_parameter('filter_uncertainty_topic', '/ibvs/filter/uncertainty')
        self.declare_parameter('filter_update_status_topic', '/ibvs/filter/update_status')
        self.declare_parameter('filter_update_count_topic', '/ibvs/filter/update_count')
        self.declare_parameter(
            'filter_update_success_count_topic',
            '/ibvs/filter/update_success_count',
        )
        self.declare_parameter('active_count_topic', '/ibvs/filter/active_count')

        self.filter_type = str(self.get_parameter('filter_type').value)
        self.q_noise = float(self.get_parameter('q_noise').value)
        self.r_noise = float(self.get_parameter('r_noise').value)
        self.gate_threshold = float(self.get_parameter('gate_threshold').value)
        self.z_depth = float(self.get_parameter('z_depth').value)
        self.predict_rate = float(self.get_parameter('predict_rate').value)

        self.max_active_keypoints = int(self.get_parameter('max_active_keypoints').value)
        self.min_init_keypoints = int(self.get_parameter('min_init_keypoints').value)
        self.min_update_keypoints = int(self.get_parameter('min_update_keypoints').value)
        self.force_relocalization_param = bool(
            self.get_parameter('force_relocalization').value
        )

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.camera_velocity_topic = str(self.get_parameter('camera_velocity_topic').value)
        self.camera_velocity_deadband_linear = float(
            self.get_parameter('camera_velocity_deadband_linear').value
        )
        self.camera_velocity_deadband_angular = float(
            self.get_parameter('camera_velocity_deadband_angular').value
        )
        self.camera_velocity_stale_timeout = float(
            self.get_parameter('camera_velocity_stale_timeout').value
        )

        self.filter_status_topic = str(self.get_parameter('filter_status_topic').value)
        self.filter_uncertainty_topic = str(self.get_parameter('filter_uncertainty_topic').value)
        self.filter_update_status_topic = str(
            self.get_parameter('filter_update_status_topic').value
        )
        self.filter_update_count_topic = str(
            self.get_parameter('filter_update_count_topic').value
        )
        self.filter_update_success_count_topic = str(
            self.get_parameter('filter_update_success_count_topic').value
        )
        self.active_count_topic = str(self.get_parameter('active_count_topic').value)

        self.K = None
        self.filter = None
        self.reference_keypoints_raw = None
        self.update_step_count = 0
        self.update_success_count = 0
        self.last_update_status = 'NO UPDATE YET'

        self.lock = threading.Lock()
        self.pose_lock = threading.Lock()

        self.latest_camera_velocity = np.zeros(6, dtype=np.float64)
        self.latest_velocity_stamp = None
        self.last_predict_time = None

        self.cb_group = ReentrantCallbackGroup()

        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.cam_info_callback,
            10,
            callback_group=self.cb_group,
        )
        self.sub_ref = self.create_subscription(
            Keypoints,
            '/ibvs/reference/keypoints',
            self.reference_callback,
            10,
            callback_group=self.cb_group,
        )
        self.sub_camera_velocity = self.create_subscription(
            Twist,
            self.camera_velocity_topic,
            self.camera_velocity_callback,
            10,
            callback_group=self.cb_group,
        )
        self.sub_matches = self.create_subscription(
            Matches,
            '/ibvs/matches',
            self.matches_callback,
            10,
            callback_group=self.cb_group,
        )

        self.pub_filtered_points = self.create_publisher(Matches, '/ibvs/filtered_features', 10)
        self.pub_filter_status = self.create_publisher(String, self.filter_status_topic, 10)
        self.pub_filter_uncertainty = self.create_publisher(
            Float32,
            self.filter_uncertainty_topic,
            10,
        )
        self.pub_filter_update_status = self.create_publisher(
            String,
            self.filter_update_status_topic,
            10,
        )
        self.pub_filter_update_count = self.create_publisher(
            UInt32,
            self.filter_update_count_topic,
            10,
        )
        self.pub_filter_update_success_count = self.create_publisher(
            UInt32,
            self.filter_update_success_count_topic,
            10,
        )
        self.pub_active_count = self.create_publisher(UInt32, self.active_count_topic, 10)

        self.timer = self.create_timer(
            1.0 / max(self.predict_rate, 1e-3),
            self.timer_callback,
            callback_group=self.cb_group,
        )

        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.get_logger().info(
            f"Filter Node gestartet. Modus: {self.filter_type}. "
            f"Warte auf K-Matrix und Referenz..."
        )

    def cam_info_callback(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.init_filter()
        self.get_logger().info('CameraInfo empfangen!')

    def init_filter(self):
        if self.filter_type == 'ekf':
            self.filter = ExtendedKalmanFilter(self.K)
        elif self.filter_type == 'ukf':
            self.filter = UnscentedKalmanFilter(self.K)
        elif self.filter_type == 'eskf':
            self.filter = ErrorStateKalmanFilter(self.K)
        elif self.filter_type == 'skf':
            self.filter = StandardKalmanFilter(self.K)
        else:
            self.get_logger().error(f'Unbekannter Filtertyp: {self.filter_type}')
            return

        self.filter.set_Q_R_gate(self.q_noise, self.r_noise, self.gate_threshold)
        self.filter.configure_keypoint_tracking(
            self.max_active_keypoints,
            self.min_init_keypoints,
            self.min_update_keypoints,
        )
        self.get_logger().info(
            'Filter initialized. '
            f'active_max={self.max_active_keypoints}, '
            f'min_init={self.min_init_keypoints}, '
            f'min_update={self.min_update_keypoints}'
        )

    def _on_parameters_changed(self, params):
        next_q = self.q_noise
        next_r = self.r_noise
        next_gate = self.gate_threshold
        next_z = self.z_depth
        next_predict_rate = self.predict_rate

        next_active_max = self.max_active_keypoints
        next_min_init = self.min_init_keypoints
        next_min_update = self.min_update_keypoints
        next_force_relocalization = self.force_relocalization_param

        next_deadband_lin = self.camera_velocity_deadband_linear
        next_deadband_ang = self.camera_velocity_deadband_angular
        next_stale_timeout = self.camera_velocity_stale_timeout
        relocalization_requested = False

        for p in params:
            if p.name == 'filter_type':
                return SetParametersResult(
                    successful=False,
                    reason='filter_type cannot be changed at runtime. Restart node.',
                )
            if p.name == 'q_noise':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='q_noise must be > 0')
                next_q = float(p.value)
            elif p.name == 'r_noise':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='r_noise must be > 0')
                next_r = float(p.value)
            elif p.name == 'gate_threshold':
                if p.value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='gate_threshold must be > 0',
                    )
                next_gate = float(p.value)
            elif p.name == 'z_depth':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='z_depth must be > 0')
                next_z = float(p.value)
            elif p.name == 'predict_rate':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='predict_rate must be > 0')
                next_predict_rate = float(p.value)
            elif p.name == 'max_active_keypoints':
                if p.value < 4:
                    return SetParametersResult(
                        successful=False,
                        reason='max_active_keypoints must be >= 4',
                    )
                next_active_max = int(p.value)
            elif p.name == 'min_init_keypoints':
                if p.value < 4:
                    return SetParametersResult(
                        successful=False,
                        reason='min_init_keypoints must be >= 4',
                    )
                next_min_init = int(p.value)
            elif p.name == 'min_update_keypoints':
                if p.value < 0:
                    return SetParametersResult(
                        successful=False,
                        reason='min_update_keypoints must be >= 0',
                    )
                next_min_update = int(p.value)
            elif p.name == 'force_relocalization':
                if not isinstance(p.value, bool):
                    return SetParametersResult(
                        successful=False,
                        reason='force_relocalization must be bool',
                    )
                next_force_relocalization = bool(p.value)
                if next_force_relocalization:
                    relocalization_requested = True
            elif p.name == 'camera_velocity_deadband_linear':
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='camera_velocity_deadband_linear must be >= 0',
                    )
                next_deadband_lin = float(p.value)
            elif p.name == 'camera_velocity_deadband_angular':
                if p.value < 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='camera_velocity_deadband_angular must be >= 0',
                    )
                next_deadband_ang = float(p.value)
            elif p.name == 'camera_velocity_stale_timeout':
                if p.value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='camera_velocity_stale_timeout must be > 0',
                    )
                next_stale_timeout = float(p.value)

        self.q_noise = next_q
        self.r_noise = next_r
        self.gate_threshold = next_gate
        self.z_depth = next_z
        self.predict_rate = next_predict_rate

        self.max_active_keypoints = next_active_max
        self.min_init_keypoints = next_min_init
        self.min_update_keypoints = next_min_update
        self.force_relocalization_param = next_force_relocalization

        self.camera_velocity_deadband_linear = next_deadband_lin
        self.camera_velocity_deadband_angular = next_deadband_ang
        self.camera_velocity_stale_timeout = next_stale_timeout

        if self.filter is not None:
            self.filter.set_Q_R_gate(self.q_noise, self.r_noise, self.gate_threshold)
            self.filter.configure_keypoint_tracking(
                self.max_active_keypoints,
                self.min_init_keypoints,
                self.min_update_keypoints,
            )
            if relocalization_requested:
                with self.lock:
                    self.filter.force_relocalization()
                self.last_update_status = 'RELOCALIZATION REQUESTED'
                self.get_logger().warn(
                    'Manual relocalization requested via parameter force_relocalization=true'
                )
        elif relocalization_requested:
            self.get_logger().warn(
                'force_relocalization requested, but filter is not initialized yet'
            )

        self.get_logger().info(
            f'Tuning updated: q={self.q_noise:.4f}, r={self.r_noise:.4f}, '
            f'gate={self.gate_threshold:.4f}, z={self.z_depth:.4f}, '
            f'active_max={self.max_active_keypoints}, '
            f'min_init={self.min_init_keypoints}, min_update={self.min_update_keypoints}'
        )

        return SetParametersResult(successful=True)

    def reference_callback(self, msg: Keypoints):
        if self.reference_keypoints_raw is None:
            self.reference_keypoints_raw = np.asarray(msg.xy, dtype=np.float64)
            self.get_logger().info(
                f'Neue Referenz empfangen! ({len(msg.xy) // 2} Keypoints)'
            )
            if self.filter is not None:
                self.filter.force_relocalization()

    def camera_velocity_callback(self, msg: Twist):
        twist = np.array(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ],
            dtype=np.float64,
        )

        twist[:3][np.abs(twist[:3]) < self.camera_velocity_deadband_linear] = 0.0
        twist[3:][np.abs(twist[3:]) < self.camera_velocity_deadband_angular] = 0.0

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        with self.pose_lock:
            self.latest_camera_velocity = twist
            self.latest_velocity_stamp = now_sec

    def get_camera_velocity_from_topic(self):
        with self.pose_lock:
            twist = self.latest_camera_velocity.copy()
            stamp = self.latest_velocity_stamp

        if stamp is None:
            return np.zeros(6, dtype=np.float64)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - stamp) > self.camera_velocity_stale_timeout:
            return np.zeros(6, dtype=np.float64)
        return twist

    def get_predict_dt(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_predict_time is None:
            self.last_predict_time = now_sec
            return 1.0 / max(float(self.predict_rate), 1e-3)

        dt = now_sec - self.last_predict_time
        self.last_predict_time = now_sec
        if dt <= 0.0 or dt > 1.0:
            return 1.0 / max(float(self.predict_rate), 1e-3)
        return dt

    def _publish_filter_meta(self, p_trace: float):
        status_msg = String()
        status_msg.data = f'{self.filter_type.upper()} | {self.filter.status}'
        self.pub_filter_status.publish(status_msg)

        unc_msg = Float32()
        unc_msg.data = float(max(0.0, p_trace))
        self.pub_filter_uncertainty.publish(unc_msg)

        upd_status_msg = String()
        upd_status_msg.data = self.last_update_status
        self.pub_filter_update_status.publish(upd_status_msg)

        upd_count_msg = UInt32()
        upd_count_msg.data = int(max(0, self.update_step_count))
        self.pub_filter_update_count.publish(upd_count_msg)

        upd_success_msg = UInt32()
        upd_success_msg.data = int(max(0, self.update_success_count))
        self.pub_filter_update_success_count.publish(upd_success_msg)

        active_count_msg = UInt32()
        if self.filter is not None:
            active_count_msg.data = int(max(0, self.filter.get_active_count()))
        else:
            active_count_msg.data = 0
        self.pub_active_count.publish(active_count_msg)

    def _extract_active_position_uncertainty(self, active_ref_ids: np.ndarray) -> np.ndarray:
        """Return per-active-keypoint position sigma in pixels from the filter covariance."""
        n = int(active_ref_ids.size)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.filter is None or (not hasattr(self.filter, 'P')):
            return np.zeros((n,), dtype=np.float32)

        p_mat = np.asarray(getattr(self.filter, 'P'), dtype=np.float64)
        if p_mat.ndim != 2:
            return np.zeros((n,), dtype=np.float32)

        sigma_px = np.zeros((n,), dtype=np.float32)
        for slot in range(n):
            i0 = 2 * slot
            i1 = i0 + 2
            if i1 <= p_mat.shape[0] and i1 <= p_mat.shape[1]:
                p_block = p_mat[i0:i1, i0:i1]
                tr = float(np.trace(p_block))
                if not np.isfinite(tr):
                    tr = 0.0
                sigma_px[slot] = float(np.sqrt(max(0.0, tr)))
            else:
                sigma_px[slot] = 0.0
        return sigma_px

    def timer_callback(self):
        if self.filter is None or self.reference_keypoints_raw is None:
            return

        dt = self.get_predict_dt()
        v_ee = self.get_camera_velocity_from_topic()

        with self.lock:
            self.filter.predict(v_ee, self.z_depth, dt)
            active_ref_ids = self.filter.get_active_ref_ids()
            filtered_current_pts = self.filter.get_active_filtered_points()
            active_sigma_px = self._extract_active_position_uncertainty(active_ref_ids)
            p_trace = float(np.trace(self.filter.P)) if hasattr(self.filter, 'P') else 0.0
            # print(f"P: {self.filter.P}, trace: {np.trace(self.filter.P):.2f}")

        out_msg = Matches()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.camera_frame

        num_pts = int(min(filtered_current_pts.shape[1], active_ref_ids.size))
        if num_pts > 0:
            out_msg.ref_id = active_ref_ids[:num_pts].astype(np.uint32).tolist()
            out_msg.xy = (
                filtered_current_pts[:, :num_pts].T.astype(np.float32).flatten().tolist()
            )
            out_msg.sim = active_sigma_px[:num_pts].astype(np.float32).tolist()
        else:
            out_msg.ref_id = []
            out_msg.xy = []
            out_msg.sim = []

        self.pub_filtered_points.publish(out_msg)
        self._publish_filter_meta(p_trace)

    def matches_callback(self, matches_msg: Matches):
        if self.filter is None or self.reference_keypoints_raw is None:
            return

        num_matches = len(matches_msg.ref_id)
        if num_matches <= 0:
            return

        current_pixels = np.zeros((2, num_matches), dtype=np.float64)
        desired_pixels = np.zeros((2, num_matches), dtype=np.float64)
        ref_ids = np.zeros((num_matches,), dtype=np.int64)
        match_scores = np.zeros((num_matches,), dtype=np.float64)

        valid_count = 0
        n_xy_pairs = len(matches_msg.xy) // 2
        n_scores = len(matches_msg.sim)

        for i, ref_idx in enumerate(matches_msg.ref_id):
            if i >= n_xy_pairs:
                break
            if (ref_idx * 2 + 1) >= len(self.reference_keypoints_raw):
                continue

            current_pixels[0, valid_count] = matches_msg.xy[i * 2]
            current_pixels[1, valid_count] = matches_msg.xy[i * 2 + 1]
            desired_pixels[0, valid_count] = self.reference_keypoints_raw[ref_idx * 2]
            desired_pixels[1, valid_count] = self.reference_keypoints_raw[ref_idx * 2 + 1]
            ref_ids[valid_count] = int(ref_idx)
            if i < n_scores:
                match_scores[valid_count] = float(matches_msg.sim[i])
            valid_count += 1

        if valid_count <= 0:
            return

        current_pixels = current_pixels[:, :valid_count]
        desired_pixels = desired_pixels[:, :valid_count]
        ref_ids = ref_ids[:valid_count]
        match_scores = match_scores[:valid_count]

        with self.lock:
            self.filter.update(
                current_pixels,
                desired_pixels,
                ref_ids=ref_ids,
                match_scores=match_scores,
            )
            update_status = str(self.filter.status)

        self.update_step_count += 1
        self.last_update_status = update_status
        if update_status in ('UPDATE', 'INIT', 'RELOCALIZED'):
            self.update_success_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = FilterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
