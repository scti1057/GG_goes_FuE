#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Twist
from controller_manager_msgs.srv import SwitchController, LoadController
from irp_ur5e_interfaces.action import ApmoveTwist  # ROS 2 generated action

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
]

CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "cartesian_twist_passthrough_controller"]


class APMoveTwist(Node):
    def __init__(self):
        super().__init__('ApmoveTwist')

        # Action server
        self._action_server = ActionServer(
            self,
            ApmoveTwist,
            'ApmoveTwist',
            execute_callback=self.execute_cb
        )

        # Frames
        self.frameBase = "base"
        self.frameTarget = "tool0_controller"

        # Twist controller
        self.twist_controller = CONFLICTING_CONTROLLERS[1]

        # Publisher for twist commands
        self.twist_pub = self.create_publisher(Twist, f'/{self.twist_controller}/cmd_vel', 10)
        self.get_logger().info(f"Connected to twist controller: {self.twist_controller}")

        # Service clients
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        if not self.load_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("LoadController service not available!")
            sys.exit(-1)
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SwitchController service not available!")
            sys.exit(-1)

    def switch_controller(self, target_controller):
        """Activate target_controller and stop all others"""
        other_controllers = JOINT_TRAJECTORY_CONTROLLERS + CARTESIAN_TRAJECTORY_CONTROLLERS + CONFLICTING_CONTROLLERS
        if target_controller in other_controllers:
            other_controllers.remove(target_controller)

        # Load controller
        load_req = LoadController.Request()
        load_req.name = target_controller
        self.load_client.call_async(load_req)
        self.get_logger().info(f"LoadController request sent for {target_controller}")

        # Switch controller
        switch_req = SwitchController.Request()
        switch_req.activate_controllers = [target_controller]
        switch_req.deactivate_controllers = other_controllers
        switch_req.strictness = switch_req.BEST_EFFORT
        switch_future = self.switch_client.call_async(switch_req)
        rclpy.spin_until_future_complete(self, switch_future, timeout_sec=5.0)
        
        if switch_future.result() is not None and switch_future.result().ok:
            self.get_logger().info(f"SwitchController succeeded for {target_controller}")
            return True
        else:
            self.get_logger().error(f"SwitchController failed for {target_controller}")
            return False

    async def execute_cb(self, goal_handle):
        """Send Twist goal to twist controller"""
        # Switch controller
        self.switch_controller(self.twist_controller)

        # Get the goal Twist
        goal_twist = goal_handle.request.twist

        # Publish Twist
        self.twist_pub.publish(goal_twist)
        self.get_logger().info("Twist command published")

        # Mark goal as succeeded
        goal_handle.succeed()
        return ApmoveTwist.Result()


def main(args=None):
    rclpy.init(args=args)
    node = APMoveTwist()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
