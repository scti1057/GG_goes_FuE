#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_ros import TransformException
class PosePublisher(Node):
    """
    ROS2 Node to publish the robot's TCP pose as a PoseStamped message
    at 500 Hz, using TF2 to lookup transform from base to tool0_controller.
    """

    def __init__(self):
        super().__init__('robot_pose_publisher')

        # Publisher for robot TCP pose
        self.pub = self.create_publisher(PoseStamped, 'robot_pose', 10)

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Frame names
        self.frame_base = 'base'
        self.frame_target = 'tool0_controller'

        # Timer for 500 Hz update rate
        timer_period = 1.0 / 500.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Pose publisher node initialized at 500 Hz')

    def timer_callback(self):
        """
        Callback function triggered by the timer at 500 Hz.
        Looks up the transform and publishes a PoseStamped message.
        """
        try:
            # Lookup the transform from base to tool
            trans = self.tf_buffer.lookup_transform(
                self.frame_base,
                self.frame_target,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"Waiting for transform: {ex}. If this persists, check TF2 frames.", throttle_duration_sec=5.0)
            return

        # Fill PoseStamped message
        tcp_pose = PoseStamped()
        tcp_pose.header.stamp = self.get_clock().now().to_msg()
        tcp_pose.header.frame_id = self.frame_base

        tcp_pose.pose.position.x = trans.transform.translation.x
        tcp_pose.pose.position.y = trans.transform.translation.y
        tcp_pose.pose.position.z = trans.transform.translation.z

        tcp_pose.pose.orientation.x = trans.transform.rotation.x
        tcp_pose.pose.orientation.y = trans.transform.rotation.y
        tcp_pose.pose.orientation.z = trans.transform.rotation.z
        tcp_pose.pose.orientation.w = trans.transform.rotation.w

        # Publish the pose
        self.pub.publish(tcp_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
