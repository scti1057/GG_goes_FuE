#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from ibvs_msgs.msg import Matches, Keypoints
from ibvs_filter.core.base import BaseFilter

"""
ros2 run ibvs_filter r_noise_estimator --ros-args \
  -p sample_count:=500 \
  -p min_matches:=4 \
  -p scalar_estimator:=median \
  -p inflation_factor:=1.5
"""


class RNoiseEstimatorNode(Node):
    def __init__(self):
        super().__init__('r_noise_estimator_node')

        self.declare_parameter('matches_topic', '/ibvs/matches')
        self.declare_parameter('reference_topic', '/ibvs/reference/keypoints')
        self.declare_parameter('sample_count', 400)
        self.declare_parameter('min_matches', 4)
        self.declare_parameter('inflation_factor', 1.5)
        self.declare_parameter('scalar_estimator', 'median')  # mean | median | p75 | max
        self.declare_parameter('output_yaml', '')

        self.matches_topic = self.get_parameter('matches_topic').value
        self.reference_topic = self.get_parameter('reference_topic').value
        self.sample_count = int(self.get_parameter('sample_count').value)
        self.min_matches = int(self.get_parameter('min_matches').value)
        self.inflation_factor = float(self.get_parameter('inflation_factor').value)
        self.scalar_estimator = str(self.get_parameter('scalar_estimator').value).strip().lower()
        self.output_yaml = str(self.get_parameter('output_yaml').value).strip()

        self.reference_keypoints_raw = None
        self.measurement_model = BaseFilter(K=None)
        self.samples = []
        self.done = False

        qos_ref = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Keypoints, self.reference_topic, self.reference_callback, qos_ref)
        self.create_subscription(Matches, self.matches_topic, self.matches_callback, 10)
        self.create_timer(0.2, self._shutdown_if_done)

        self.get_logger().info(
            f"r_noise estimator running. Waiting for reference on '{self.reference_topic}' "
            f"and matches on '{self.matches_topic}'."
        )

    def reference_callback(self, msg: Keypoints):
        new_ref = np.array(msg.xy, dtype=np.float64)
        if self.reference_keypoints_raw is None or len(new_ref) != len(self.reference_keypoints_raw):
            self.samples.clear()
            self.measurement_model.reset()
            self.get_logger().info("Reference received/changed: sample buffer reset.")
        self.reference_keypoints_raw = new_ref

    def matches_callback(self, msg: Matches):
        if self.done or self.reference_keypoints_raw is None:
            return

        num_matches = len(msg.ref_id)
        if num_matches < self.min_matches:
            return

        current_pixels = np.zeros((2, num_matches), dtype=np.float64)
        desired_pixels = np.zeros((2, num_matches), dtype=np.float64)

        valid_count = 0
        ref_len = len(self.reference_keypoints_raw)
        for i, ref_idx in enumerate(msg.ref_id):
            ref_pos = int(ref_idx) * 2
            if ref_pos + 1 < ref_len and (i * 2 + 1) < len(msg.xy):
                current_pixels[0, valid_count] = msg.xy[i * 2]
                current_pixels[1, valid_count] = msg.xy[i * 2 + 1]
                desired_pixels[0, valid_count] = self.reference_keypoints_raw[ref_pos]
                desired_pixels[1, valid_count] = self.reference_keypoints_raw[ref_pos + 1]
                valid_count += 1

        if valid_count < self.min_matches:
            return

        current_pixels = current_pixels[:, :valid_count]
        desired_pixels = desired_pixels[:, :valid_count]

        z_k, _, _ = self.measurement_model._get_raw_measurement(current_pixels, desired_pixels)
        if z_k is None:
            return

        z = z_k.reshape(-1).astype(np.float64)
        if z.shape[0] != 8 or not np.isfinite(z).all():
            return

        self.samples.append(z)
        n = len(self.samples)
        if n % 50 == 0 or n == self.sample_count:
            self.get_logger().info(f"Collected {n}/{self.sample_count} valid samples.")

        if n >= self.sample_count:
            self._finalize()

    def _finalize(self):
        Z = np.vstack(self.samples)
        cov = np.cov(Z, rowvar=False, ddof=1) if Z.shape[0] > 1 else np.zeros((8, 8), dtype=np.float64)
        diag = np.clip(np.diag(cov), 0.0, None)

        scalar_map = {
            'mean': float(np.mean(diag)),
            'median': float(np.median(diag)),
            'p75': float(np.percentile(diag, 75.0)),
            'max': float(np.max(diag)),
        }
        if self.scalar_estimator not in scalar_map:
            self.get_logger().warn(
                f"Unknown scalar_estimator='{self.scalar_estimator}', fallback to 'median'."
            )
            self.scalar_estimator = 'median'

        base_scalar = scalar_map[self.scalar_estimator]
        recommended = max(base_scalar * self.inflation_factor, 1e-6)

        self.get_logger().info(
            "Estimated R statistics (from z_k covariance diag):\n"
            f"diag = {np.array2string(diag, precision=3, separator=', ')}\n"
            f"mean={scalar_map['mean']:.6f}, median={scalar_map['median']:.6f}, "
            f"p75={scalar_map['p75']:.6f}, max={scalar_map['max']:.6f}\n"
            f"chosen='{self.scalar_estimator}', inflation={self.inflation_factor:.3f}\n"
            f"recommended r_noise = {recommended:.6f}"
        )
        self.get_logger().info(
            f"Command: ros2 param set /ibvs_filter_node r_noise {recommended:.6f}"
        )

        if self.output_yaml:
            self._write_yaml(recommended)

        self.done = True

    def _write_yaml(self, r_noise_value: float):
        content = (
            "ibvs_filter_node:\n"
            "  ros__parameters:\n"
            f"    r_noise: {r_noise_value:.6f}\n"
        )
        with open(self.output_yaml, 'w', encoding='ascii') as f:
            f.write(content)
        self.get_logger().info(f"Wrote recommendation YAML to: {self.output_yaml}")

    def _shutdown_if_done(self):
        if self.done:
            self.get_logger().info("r_noise estimation finished. Shutting down.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RNoiseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
