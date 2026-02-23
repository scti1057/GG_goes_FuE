#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile
import numpy as np
from geometry_msgs.msg import Twist, WrenchStamped
from controller_manager_msgs.srv import SwitchController, LoadController
from irp_ur5e_interfaces.action import AdmittanzRegler

class AdmittanzReglerServer(Node):
    def __init__(self):
        super().__init__('admittanz_regler_server')

        # Action server
        self._action_server = ActionServer(
            self,
            AdmittanzRegler,
            'admittanz_regler',
            execute_callback=self.execute_cb
        )

        # Activation mask and controller parameters
        self.activation_mask = np.array([1.0, -1.0, 1.0, 0, 0, 0])
        self.M = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
        self.D = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])
        self.C = np.array([40.0, 40.0, 40.0, 1.0, 1.0, 1.0])

        self.vel_limit_linear = 0.02
        self.vel_limit_angular = 0.02
        self.Hz = 50
        self.T = 1 / self.Hz

        self.calculated_twist = np.zeros(6)
        self.target_wrench = np.zeros(6)
        self.actual_wrench = np.zeros(6)

        # Subscriber to force/torque sensor
        qos = QoSProfile(depth=10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/external_force_torque_broadcaster/wrench', # New ros2 topic name as the sensor is now published by external_force_torque_broadcaster node launched by ros control
            self.wrench_cb,
            qos
        )

        # Publisher to the controller
        self.controller_pub = self.create_publisher(Twist, '/cartesian_twist_passthrough_controller/cmd_vel', qos)

        # Service clients for controller management
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')

        # Controller lists
        self.JOINT_TRAJECTORY_CONTROLLERS = [
            "scaled_pos_joint_traj_controller",
            "scaled_vel_joint_traj_controller",
            "pos_joint_traj_controller",
            "vel_joint_traj_controller",
            "forward_joint_traj_controller",
        ]
        self.CARTESIAN_TRAJECTORY_CONTROLLERS = [
            "pose_based_cartesian_traj_controller",
            "joint_based_cartesian_traj_controller",
            "forward_cartesian_traj_controller",
            "my_cartesian_force_controller",
        ]
        self.CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "cartesian_twist_passthrough_controller"]

        # Switch to initial controller
        self.switch_controller('cartesian_twist_passthrough_controller')

    def execute_cb(self, goal_handle):
        self.get_logger().info('Action started')
        rate = self.create_rate(self.Hz)
        success = True

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.controller_pub.publish(Twist())  # Stop motion
                goal_handle.canceled()
                success = False
                break

            self.target_wrench = self.wrench2vec(goal_handle.request.wrench)
            self.error_wrench = self.error_threshold()
            self.calculated_twist = self.select_axes(self.error_wrench)
            twist_msg = self.vec2twist(self.calculated_twist)
            self.controller_pub.publish(twist_msg)
            rate.sleep()

        if success:
            self.controller_pub.publish(Twist())
            goal_handle.succeed()
            result = AdmittanzRegler.Result()
            result.success = True
            return result

    def wrench_cb(self, msg):
        self.actual_wrench = self.wrench2vec(msg.wrench)

    def wrench2vec(self, wrench):
        return np.array([
            wrench.force.x, wrench.force.y, wrench.force.z,
            wrench.torque.x, wrench.torque.y, wrench.torque.z
        ])

    def vec2twist(self, vec):
        twist = Twist()
        twist.linear.x = vec[0]
        twist.linear.y = vec[1]
        twist.linear.z = vec[2]
        twist.angular.x = vec[3]
        twist.angular.y = vec[4]
        twist.angular.z = vec[5]
        return twist

    def select_axes(self, twist):
        return self.activation_mask * twist

    def error_threshold(self):
        error = self.target_wrench - self.actual_wrench
        mask = np.abs(error) > 0.8
        return error * mask

    def switch_controller(self, target_controller):
        other_controllers = (
            self.JOINT_TRAJECTORY_CONTROLLERS +
            self.CARTESIAN_TRAJECTORY_CONTROLLERS +
            self.CONFLICTING_CONTROLLERS
        )
        if target_controller in other_controllers:
            other_controllers.remove(target_controller)

        # Load controller
        load_req = LoadController.Request()
        load_req.name = target_controller
        if self.load_client.wait_for_service(timeout_sec=5.0):
            self.load_client.call_async(load_req)

        # Switch controller
        switch_req = SwitchController.Request()
        switch_req.activate_controllers = [target_controller]
        switch_req.deactivate_controllers = other_controllers
        switch_req.strictness = SwitchController.Request.BEST_EFFORT
        if self.switch_client.wait_for_service(timeout_sec=5.0):
            self.switch_client.call_async(switch_req)


def main(args=None):
    rclpy.init(args=args)
    server = AdmittanzReglerServer()
    rclpy.spin(server)
    server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()