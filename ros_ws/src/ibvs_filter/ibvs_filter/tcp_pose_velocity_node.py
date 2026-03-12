#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node


class TcpPoseVelocityNode(Node):
    def __init__(self):
        super().__init__('tcp_pose_velocity_node')

        self.declare_parameter('pose_topic', '/tcp_pose_broadcaster/pose')
        self.declare_parameter('twist_stamped_topic', '/tcp_pose_broadcaster/velocity')
        self.declare_parameter('twist_topic', '/tcp_pose_broadcaster/velocity_unstamped')
        self.declare_parameter('publish_unstamped_twist', False)
        self.declare_parameter('use_message_stamp', True)
        self.declare_parameter('min_dt_sec', 1.0e-4)
        self.declare_parameter('linear_deadband_mps', 0.003)
        self.declare_parameter('angular_deadband_radps', 0.03)
        self.declare_parameter('enable_ema', True)
        self.declare_parameter('ema_alpha', 0.2)

        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.twist_stamped_topic = str(self.get_parameter('twist_stamped_topic').value)
        self.twist_topic = str(self.get_parameter('twist_topic').value)
        self.publish_unstamped_twist = bool(
            self.get_parameter('publish_unstamped_twist').value
        )
        self.use_message_stamp = bool(self.get_parameter('use_message_stamp').value)
        self.min_dt_sec = float(self.get_parameter('min_dt_sec').value)
        self.linear_deadband_mps = float(self.get_parameter('linear_deadband_mps').value)
        self.angular_deadband_radps = float(
            self.get_parameter('angular_deadband_radps').value
        )
        self.enable_ema = bool(self.get_parameter('enable_ema').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)

        if not (0.0 < self.ema_alpha <= 1.0):
            self.get_logger().warn(
                f"Invalid ema_alpha={self.ema_alpha}. Falling back to 1.0 (no smoothing)."
            )
            self.ema_alpha = 1.0

        self.prev_position: Optional[np.ndarray] = None
        self.prev_orientation: Optional[np.ndarray] = None
        self.prev_time_sec: Optional[float] = None
        self.prev_frame_id: Optional[str] = None
        self.filtered_linear_velocity: Optional[np.ndarray] = None
        self.filtered_angular_velocity: Optional[np.ndarray] = None

        self.sub_pose = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )
        self.pub_twist_stamped = self.create_publisher(
            TwistStamped,
            self.twist_stamped_topic,
            10,
        )
        self.pub_twist = None
        if self.publish_unstamped_twist:
            self.pub_twist = self.create_publisher(Twist, self.twist_topic, 10)

        self.get_logger().info(
            f"Sub pose={self.pose_topic} "
            f"Pub twist_stamped={self.twist_stamped_topic} "
            f"use_message_stamp={self.use_message_stamp} "
            f"deadband_lin={self.linear_deadband_mps:.4f} "
            f"deadband_ang={self.angular_deadband_radps:.4f} "
            f"ema={'on' if self.enable_ema else 'off'} alpha={self.ema_alpha:.2f}"
        )
        if self.publish_unstamped_twist:
            self.get_logger().info(f"Pub twist={self.twist_topic}")

    def pose_callback(self, msg: PoseStamped):
        now_sec = self._extract_time_sec(msg)
        if now_sec is None:
            return

        position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )

        orientation = self._normalize_quaternion(
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )

        if (
            self.prev_position is None
            or self.prev_orientation is None
            or self.prev_time_sec is None
        ):
            self.prev_position = position
            self.prev_orientation = orientation
            self.prev_time_sec = now_sec
            self.prev_frame_id = msg.header.frame_id
            self.filtered_linear_velocity = None
            self.filtered_angular_velocity = None
            return

        dt = now_sec - self.prev_time_sec
        if dt < self.min_dt_sec:
            return

        if (
            self.prev_frame_id
            and msg.header.frame_id
            and msg.header.frame_id != self.prev_frame_id
        ):
            self.get_logger().warn(
                f"Frame changed from '{self.prev_frame_id}' to '{msg.header.frame_id}'. "
                "Resetting differentiator."
            )
            self.prev_position = position
            self.prev_orientation = orientation
            self.prev_time_sec = now_sec
            self.prev_frame_id = msg.header.frame_id
            self.filtered_linear_velocity = None
            self.filtered_angular_velocity = None
            return

        linear_velocity = (position - self.prev_position) / dt
        angular_velocity = self._compute_angular_velocity(
            self.prev_orientation,
            orientation,
            dt,
        )
        linear_velocity, angular_velocity = self._apply_deadband(
            linear_velocity,
            angular_velocity,
        )
        linear_velocity, angular_velocity = self._apply_ema(
            linear_velocity,
            angular_velocity,
        )

        twist_stamped = TwistStamped()
        twist_stamped.header = msg.header
        twist_stamped.twist.linear.x = float(linear_velocity[0])
        twist_stamped.twist.linear.y = float(linear_velocity[1])
        twist_stamped.twist.linear.z = float(linear_velocity[2])
        twist_stamped.twist.angular.x = float(angular_velocity[0])
        twist_stamped.twist.angular.y = float(angular_velocity[1])
        twist_stamped.twist.angular.z = float(angular_velocity[2])
        self.pub_twist_stamped.publish(twist_stamped)

        if self.pub_twist is not None:
            self.pub_twist.publish(twist_stamped.twist)

        self.prev_position = position
        self.prev_orientation = orientation
        self.prev_time_sec = now_sec
        self.prev_frame_id = msg.header.frame_id

    def _extract_time_sec(self, msg: PoseStamped) -> Optional[float]:
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if self.use_message_stamp and stamp_sec > 0.0:
            return stamp_sec
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quaternion)
        if norm < 1.0e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quaternion / norm

    @staticmethod
    def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
        return np.array(
            [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lx, ly, lz, lw = lhs
        rx, ry, rz, rw = rhs
        return np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )

    def _compute_angular_velocity(
        self,
        previous_orientation: np.ndarray,
        current_orientation: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if float(np.dot(previous_orientation, current_orientation)) < 0.0:
            current_orientation = -current_orientation

        delta_quaternion = self._quat_multiply(
            self._quat_conjugate(previous_orientation),
            current_orientation,
        )
        delta_quaternion = self._normalize_quaternion(delta_quaternion)

        delta_vector = delta_quaternion[:3]
        delta_norm = float(np.linalg.norm(delta_vector))
        if delta_norm < 1.0e-12:
            return np.zeros(3, dtype=np.float64)

        angle = 2.0 * math.atan2(delta_norm, float(delta_quaternion[3]))
        if angle > math.pi:
            angle -= 2.0 * math.pi

        axis = delta_vector / delta_norm
        return axis * (angle / dt)

    def _apply_deadband(
        self,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.linalg.norm(linear_velocity) < self.linear_deadband_mps:
            linear_velocity = np.zeros(3, dtype=np.float64)
        if np.linalg.norm(angular_velocity) < self.angular_deadband_radps:
            angular_velocity = np.zeros(3, dtype=np.float64)
        return linear_velocity, angular_velocity

    def _apply_ema(
        self,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_ema:
            return linear_velocity, angular_velocity

        if self.filtered_linear_velocity is None:
            self.filtered_linear_velocity = linear_velocity
        else:
            self.filtered_linear_velocity = (
                self.ema_alpha * linear_velocity
                + (1.0 - self.ema_alpha) * self.filtered_linear_velocity
            )

        if self.filtered_angular_velocity is None:
            self.filtered_angular_velocity = angular_velocity
        else:
            self.filtered_angular_velocity = (
                self.ema_alpha * angular_velocity
                + (1.0 - self.ema_alpha) * self.filtered_angular_velocity
            )

        return self.filtered_linear_velocity, self.filtered_angular_velocity


def main(args: Optional[Tuple[str, ...]] = None):
    rclpy.init(args=args)
    node = TcpPoseVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
