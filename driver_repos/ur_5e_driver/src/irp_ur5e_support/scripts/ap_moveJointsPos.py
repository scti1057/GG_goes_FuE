#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, LoadController
from irp_ur5e_support.msg import ap_moveJointsPos  # Assuming ROS 2 IDL generated messages

# Robot joint names
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

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


class APMoveJointsPos(Node):
    def __init__(self):
        super().__init__('ap_moveJointsPos')

        # Action server
        self._action_server = ActionServer(
            self,
            ap_moveJointsPos,  # ROS 2 generated action type
            'ap_moveJointsPos',
            execute_callback=self.execute_cb
        )

        # Trajectory execution parameters
        self.traj_time = 10.0
        self.joint_trajectory_controller = JOINT_TRAJECTORY_CONTROLLERS[2]

        # Action client to controller
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            f'{self.joint_trajectory_controller}/follow_joint_trajectory'
        )

        self.get_logger().info(f"Connecting to controller action server: {self.joint_trajectory_controller}")
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Could not reach controller action server")
            sys.exit(-1)
        self.get_logger().info("Connected to controller action server")

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
        switch_req.start_controllers = [target_controller]
        switch_req.stop_controllers = other_controllers
        switch_req.strictness = switch_req.BEST_EFFORT
        self.switch_client.call_async(switch_req)
        self.get_logger().info(f"SwitchController request sent for {target_controller}")

    async def execute_cb(self, goal_handle):
        """Execute joint trajectory action"""
        self.switch_controller(self.joint_trajectory_controller)

        # Construct trajectory goal
        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.trajectory.joint_names = JOINT_NAMES
        trajectory_goal.trajectory.points.append(goal_handle.request.point)  # Assuming request has point field

        # Send trajectory goal
        self.get_logger().info("Sending trajectory goal to controller")
        send_goal_future = self.trajectory_client.send_goal_async(trajectory_goal)
        await send_goal_future
        result_future = self.trajectory_client.get_result_async()
        result = await result_future

        self.get_logger().info(f"Trajectory execution finished with status: {result.status}")
        goal_handle.succeed()
        return ap_moveJointsPos.Result()

def main(args=None):
    rclpy.init(args=args)
    node = APMoveJointsPos()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
