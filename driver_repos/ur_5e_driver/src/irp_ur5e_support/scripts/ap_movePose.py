#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.action import ActionClientfrom geometry_msgs.msg import Pose
from controller_manager_msgs.srv import SwitchController, LoadController
from cartesian_control_msgs.action import FollowCartesianTrajectory, CartesianTrajectoryPoint
from irp_ur5e_support.msg import ap_movePose  # ROS 2 generated action
import numpy as np

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

CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]


class APMovePose(Node):
    def __init__(self):
        super().__init__('ap_movePose')

        # Action server
        self._action_server = ActionServer(
            self,
            ap_movePose,  # ROS 2 IDL action type
            'ap_movePose',
            execute_callback=self.execute_cb
        )

        # Frames
        self.frameBase = "base"
        self.frameTarget = "tool0_controller"

        # Trajectory options
        self.steps = 10
        self.traj_time = 5.0

        # Cartesian controller
        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[2]

        # Action client to controller
        self.trajectory_client = ActionClient(
            self,
            FollowCartesianTrajectory,
            f'{self.cartesian_trajectory_controller}/follow_cartesian_trajectory'
        )

        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Could not reach Cartesian controller action server")
            sys.exit(-1)
        self.get_logger().info(f"Connected to Cartesian controller: {self.cartesian_trajectory_controller}")

        # Service clients
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        if not self.load_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("LoadController service not available!")
            sys.exit(-1)

        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SwitchController service not available!")
            sys.exit(-1)

        self.switch_controller(self.cartesian_trajectory_controller)

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
        switch_req.start_controllers = [target_controller]
        switch_req.stop_controllers = other_controllers
        switch_req.strictness = switch_req.BEST_EFFORT
        self.switch_client.call_async(switch_req)
        self.get_logger().info(f"SwitchController request sent for {target_controller}")

    async def execute_cb(self, goal_handle):
        """Execute Cartesian trajectory action"""
        self.switch_controller(self.cartesian_trajectory_controller)

        goal_pose = goal_handle.request.pose
        # Create trajectory goal
        traj_goal = FollowCartesianTrajectory.Goal()
        traj_goal.trajectory.controlled_frame = self.frameTarget
        traj_goal.trajectory.header.frame_id = self.frameBase

        point = CartesianTrajectoryPoint()
        point.pose = goal_pose
        point.time_from_start.sec = int(self.traj_time)
        traj_goal.trajectory.points.append(point)

        # Send trajectory goal
        self.get_logger().info(f"Executing Cartesian trajectory to position: {goal_pose.position}, orientation: {goal_pose.orientation}")
        send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        await send_goal_future
        result_future = self.trajectory_client.get_result_async()
        result = await result_future

        self.get_logger().info(f"Trajectory execution finished with status: {result.status}")
        goal_handle.succeed()
        return ap_movePose.Result()

def main(args=None):
    rclpy.init(args=args)
    node = APMovePose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
