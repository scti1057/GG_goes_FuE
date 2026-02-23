#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import WrenchStamped
import tf2_ros
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
import time
from controller_manager_msgs.srv import SwitchController, LoadController

# Desired force
target_force = -10.0

# Controllers
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]

CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
    "my_cartesian_force_controller",
]

CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]

force_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[3]
traj_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[2]

# Global state variables
reached_desired_force = False
z_force = 0.0


def force_feedback_callback(msg):
    global reached_desired_force, z_force
    z_force = msg.wrench.force.z
    print(f"Force feedback: {z_force}")
    if z_force <= target_force:
        reached_desired_force = True
        print("Target force reached!")


def switch_controller(node, target_controller):
    """Activate target_controller and stop all others."""
    other_controllers = JOINT_TRAJECTORY_CONTROLLERS + CARTESIAN_TRAJECTORY_CONTROLLERS + CONFLICTING_CONTROLLERS
    if target_controller in other_controllers:
        other_controllers.remove(target_controller)

    # Load controller
    load_client = node.create_client(LoadController, '/controller_manager/load_controller')
    load_req = LoadController.Request()
    load_req.name = target_controller
    if load_client.wait_for_service(timeout_sec=5.0):
        load_client.call_async(load_req)
        node.get_logger().info(f'Load controller request sent: {target_controller}')
    else:
        node.get_logger().warn('LoadController service not available!')

    # Switch controller
    switch_client = node.create_client(SwitchController, '/controller_manager/switch_controller')
    switch_req = SwitchController.Request()
    switch_req.start_controllers = [target_controller]
    switch_req.stop_controllers = other_controllers
    switch_req.strictness = switch_req.BEST_EFFORT
    if switch_client.wait_for_service(timeout_sec=5.0):
        switch_client.call_async(switch_req)
        node.get_logger().info(f'Switch controller request sent: {target_controller}')
    else:
        node.get_logger().warn('SwitchController service not available!')


def subscribe_force_feedback(node):
    """Subscribe to the force sensor topic"""
    qos = QoSProfile(depth=10)
    node.create_subscription(
        WrenchStamped,
        '/wrench',
        force_feedback_callback,
        qos
    )
    print('Subscribed to /wrench')


def publish():
    global reached_desired_force

    print("Script launched")

    # Initialize ROS 2
    rclpy.init()
    node = Node("force_controller_test")

    # TF listener (if needed for transforms)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    # Publisher
    qos = QoSProfile(depth=10)
    force_pub = node.create_publisher(
        WrenchStamped,
        '/my_cartesian_force_controller/target_wrench',
        qos
    )

    # Subscribe to force feedback
    subscribe_force_feedback(node)

    # Switch to force controller
    print('Switching Controller')
    switch_controller(node, force_controller)

    # Create force goal message
    force_goal = WrenchStamped()
    force_goal.wrench.force.x = 0.0
    force_goal.wrench.force.y = 0.0
    force_goal.wrench.force.z = target_force
    force_goal.wrench.torque.x = 0.0
    force_goal.wrench.torque.y = 0.0
    force_goal.wrench.torque.z = 0.0

    print('Publishing force goal...')
    reached_desired_force = False
    while rclpy.ok() and not reached_desired_force:
        force_pub.publish(force_goal)
        print('Force goal published, waiting for feedback...')
        rclpy.spin_once(node, timeout_sec=1.0)
        time.sleep(0.1)

    print('Switching back to trajectory controller')
    switch_controller(node, traj_controller)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        publish()
    except KeyboardInterrupt:
        print("Interrupted, switching back to trajectory controller")
        rclpy.init()
        node = Node("force_controller_cleanup")
        switch_controller(node, traj_controller)
        node.destroy_node()
        rclpy.shutdown()
