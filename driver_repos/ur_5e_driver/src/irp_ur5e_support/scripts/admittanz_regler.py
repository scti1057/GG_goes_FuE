#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from geometry_msgs.msg import WrenchStamped, Twist
from controller_manager_msgs.srv import SwitchController, LoadController
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle

target_force = -10.0

class AdmittanzRegler(Node):
    def __init__(self):
        super().__init__('admittanz_regler')

        # Freiheitsgrade aktivieren(1)/deaktivieren(0) [x, y, z, u, v, w]
        self.activation_mask = np.array([0.0, 0.0, 1.0, 0, 0, 0])

        # Reglerparameter
        self.M = np.array([200.0, 200.0, 100.0, 200.0, 200.0, 200.0])
        self.C = np.array([50.0, 50.0, 80.0, 50.0, 50.0, 50.0])
        self.D = 2.828427 * 2 * np.sqrt(self.M * self.C)

        # Geschwindigkeitlimits
        self.vel_limit_linear = 0.02
        self.vel_limit_angular = 0.02

        # Sampling time
        self.Hz = 50
        self.T = 1 / self.Hz

        # Initialize states
        self.calculated_twist = np.zeros(6)
        self.target_wrench = np.zeros(6)
        self.actual_wrench = np.zeros(6)

        # Controllers
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

        self.CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]

        # Publisher for controller
        qos = QoSProfile(depth=10)
        self.controller_pub = self.create_publisher(Twist, '/twist_controller/command', qos)

        # Subscriber to FT sensor
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/ftn_axia',
            self.wrench_cb,
            qos
        )

        # Subscriber for desired wrench
        self.goal_sub = self.create_subscription(
            WrenchStamped,
            '/admittanz_regler/desired_wrench',
            self.goal_cb,
            qos
        )

        # Service clients
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')

        # Switch to initial controller
        self.switch_controller("twist_controller")

        # Start control loop
        self.control_loop()

    def control_loop(self):
        rate_sec = 1.0 / self.Hz
        while rclpy.ok():
            # Compute error and apply admittance control
            error = self.target_wrench - self.actual_wrench
            self.error_wrench = self.error_threshold(error)
            self.calculated_twist = self.error_wrench / self.C

            # Activate/deactivate axes
            twist_vec = self.select_axes(self.calculated_twist)

            # Clip twist
            twist_vec[0:3] = np.clip(twist_vec[0:3], -self.vel_limit_linear, self.vel_limit_linear)
            twist_vec[3:6] = np.clip(twist_vec[3:6], -self.vel_limit_angular, self.vel_limit_angular)

            # Publish twist
            self.controller_pub.publish(self.vec2twist(twist_vec))

            rclpy.spin_once(self, timeout_sec=rate_sec)

    def wrench_cb(self, msg: WrenchStamped):
        self.actual_wrench = self.wrench2vec(msg.wrench)

    def goal_cb(self, msg: WrenchStamped):
        self.target_wrench = self.wrench2vec(msg.wrench)

    def wrench2vec(self, wrench):
        return np.array([
            wrench.force.x, wrench.force.y, wrench.force.z,
            wrench.torque.x, wrench.torque.y, wrench.torque.z
        ])

    def vec2twist(self, vec):
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = vec[0:3]
        twist.angular.x, twist.angular.y, twist.angular.z = vec[3:6]
        return twist

    def select_axes(self, twist_vec):
        return self.activation_mask * twist_vec

    def error_threshold(self, error):
        mask = np.abs(error) > 0.01
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
        if self.load_client.wait_for_service(timeout_sec=5.0):
            req = LoadController.Request()
            req.name = target_controller
            self.load_client.call_async(req)
            self.get_logger().info(f"LoadController request sent for {target_controller}")
        else:
            self.get_logger().error("Failed to connect to load_client service")
