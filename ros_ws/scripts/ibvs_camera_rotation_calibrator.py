#!/usr/bin/env python3

import argparse
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros import Buffer, TransformException, TransformListener


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return 3x3 rotation matrix from quaternion (xyzw)."""
    n = x * x + y * y + z * z + w * w
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n

    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


class CameraRotationCalibrator(Node):
    """Interactive camera-rotation calibrator using TF-based camera->tcp twist conversion."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("ibvs_camera_rotation_calibrator")

        self.declare_parameter("twist_topic", args.twist_topic)
        self.declare_parameter("publish_rate_hz", float(args.publish_rate_hz))
        self.declare_parameter("linear_speed_m_s", float(args.linear_speed_m_s))
        self.declare_parameter("angular_speed_rad_s", float(args.angular_speed_rad_s))
        self.declare_parameter("base_frame", args.base_frame)
        self.declare_parameter("tcp_frame", args.tcp_frame)
        self.declare_parameter("camera_frame", args.camera_frame)
        self.declare_parameter("tf_lookup_timeout_sec", float(args.tf_lookup_timeout_sec))
        self.declare_parameter("command_frame_mode", args.command_frame_mode)

        topic = str(self.get_parameter("twist_topic").value)
        self.pub = self.create_publisher(Twist, topic, 10)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._lock = threading.Lock()
        self._active_cmd: Optional[str] = None

        self._tf_ready_tcp_cam = False
        self._tf_ready_base_tcp = False
        self._R_tcp_cam = np.eye(3, dtype=np.float64)
        self._p_tcp_cam = np.zeros((3,), dtype=np.float64)
        self._R_base_tcp = np.eye(3, dtype=np.float64)

        self._last_tf_warn_sec = 0.0

        rate_hz = max(1e-3, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self._tf_timer = self.create_timer(0.5, self._update_tf_cache)

        self.get_logger().info(f"Publishing calibration twists to: {topic}")
        self.get_logger().info(
            f"TF config: base_frame={self._base_frame()} tcp_frame={self._tcp_frame()} camera_frame={self._camera_frame()}"
        )
        self.get_logger().info(f"Command mode: {self._mode()}")

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _tcp_frame(self) -> str:
        return str(self.get_parameter("tcp_frame").value)

    def _base_frame(self) -> str:
        return str(self.get_parameter("base_frame").value)

    def _camera_frame(self) -> str:
        return str(self.get_parameter("camera_frame").value)

    def _mode(self) -> str:
        mode = str(self.get_parameter("command_frame_mode").value).strip().lower()
        return "camera" if mode != "tcp" else "tcp"

    @staticmethod
    def _build_source_twist(cmd: Optional[str], linear: float, angular: float) -> np.ndarray:
        # [vx, vy, vz, wx, wy, wz] in selected source frame (camera or tcp).
        v6 = np.zeros((6,), dtype=np.float64)
        if cmd == "tx":
            v6[0] = linear
        elif cmd == "ty":
            v6[1] = linear
        elif cmd == "tz":
            v6[2] = linear
        elif cmd == "rx":
            v6[3] = angular
        elif cmd == "ry":
            v6[4] = angular
        elif cmd == "rz":
            v6[5] = angular
        return v6

    def _update_tf_cache(self):
        timeout_sec = max(1e-3, float(self.get_parameter("tf_lookup_timeout_sec").value))

        # base <- tcp (needed for publishing command in base frame)
        try:
            tf_base_tcp = self.tf_buffer.lookup_transform(
                self._base_frame(),
                self._tcp_frame(),
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
            q_bt = tf_base_tcp.transform.rotation
            R_base_tcp = quat_to_rotmat(q_bt.x, q_bt.y, q_bt.z, q_bt.w)

            with self._lock:
                first_ready = not self._tf_ready_base_tcp
                self._R_base_tcp = R_base_tcp
                self._tf_ready_base_tcp = True

            if first_ready:
                self.get_logger().info(
                    f"TF ready ({self._base_frame()} <- {self._tcp_frame()})"
                )
        except TransformException as exc:
            now = self._now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"TF lookup failed ({self._base_frame()} <- {self._tcp_frame()}): {exc}"
                )

        # tcp <- camera (needed for camera-frame command mode)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self._tcp_frame(),
                self._camera_frame(),
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except TransformException as exc:
            now = self._now_sec()
            if (now - self._last_tf_warn_sec) > 1.0:
                self._last_tf_warn_sec = now
                self.get_logger().warn(
                    f"TF lookup failed ({self._tcp_frame()} <- {self._camera_frame()}): {exc}"
                )
            return

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        p = np.array([t.x, t.y, t.z], dtype=np.float64)
        R = quat_to_rotmat(q.x, q.y, q.z, q.w)

        with self._lock:
            self._p_tcp_cam = p
            self._R_tcp_cam = R
            first_ready = not self._tf_ready_tcp_cam
            self._tf_ready_tcp_cam = True

        if first_ready:
            self.get_logger().info(
                f"TF ready ({self._tcp_frame()} <- {self._camera_frame()}), "
                f"p_tcp_cam=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]"
            )

    def _cam_twist_to_tcp_twist(self, cam_v6: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            if not self._tf_ready_tcp_cam:
                return None
            R = self._R_tcp_cam.copy()
            p = self._p_tcp_cam.copy()

        v_cam = cam_v6[0:3]
        w_cam = cam_v6[3:6]

        # Express angular/linear camera-point twist in tcp frame.
        w_tcp = R @ w_cam
        v_cam_point_tcp = R @ v_cam

        # Spatial velocity shift from camera point to tcp origin:
        # v_cam_point = v_tcp + w_tcp x p_tcp_cam  -> v_tcp = v_cam_point - w_tcp x p_tcp_cam
        v_tcp = v_cam_point_tcp - np.cross(w_tcp, p)

        out = np.zeros((6,), dtype=np.float64)
        out[0:3] = v_tcp
        out[3:6] = w_tcp
        return out

    def _tcp_twist_to_base_twist(self, tcp_v6: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            if not self._tf_ready_base_tcp:
                return None
            R = self._R_base_tcp.copy()

        out = np.zeros((6,), dtype=np.float64)
        out[0:3] = R @ tcp_v6[0:3]
        out[3:6] = R @ tcp_v6[3:6]
        return out

    @staticmethod
    def _to_twist_msg(v6: np.ndarray) -> Twist:
        msg = Twist()
        msg.linear.x = float(v6[0])
        msg.linear.y = float(v6[1])
        msg.linear.z = float(v6[2])
        msg.angular.x = float(v6[3])
        msg.angular.y = float(v6[4])
        msg.angular.z = float(v6[5])
        return msg

    def _on_timer(self):
        with self._lock:
            cmd = self._active_cmd

        linear = float(self.get_parameter("linear_speed_m_s").value)
        omega = float(self.get_parameter("angular_speed_rad_s").value)
        source_v6 = self._build_source_twist(cmd, linear, omega)

        if self._mode() == "tcp":
            tcp_v6 = source_v6
        else:
            tcp_v6 = self._cam_twist_to_tcp_twist(source_v6)
            if tcp_v6 is None:
                self.pub.publish(Twist())
                return

        base_v6 = self._tcp_twist_to_base_twist(tcp_v6)
        if base_v6 is None:
            self.pub.publish(Twist())
            return

        self.pub.publish(self._to_twist_msg(base_v6))

    def set_command(self, cmd: Optional[str]):
        with self._lock:
            self._active_cmd = cmd

        linear = float(self.get_parameter("linear_speed_m_s").value)
        omega = float(self.get_parameter("angular_speed_rad_s").value)
        source_v6 = self._build_source_twist(cmd, linear, omega)
        cmd_txt = cmd if cmd is not None else "none"

        if self._mode() == "camera":
            tcp_v6 = self._cam_twist_to_tcp_twist(source_v6)
            if tcp_v6 is None:
                self.get_logger().warn(
                    f"Switched command -> {cmd_txt}, but TF not ready yet "
                    f"({self._tcp_frame()} <- {self._camera_frame()})."
                )
                return
        else:
            tcp_v6 = source_v6

        if self._tcp_twist_to_base_twist(tcp_v6) is None:
            self.get_logger().warn(
                f"Switched command -> {cmd_txt}, but TF not ready yet "
                f"({self._base_frame()} <- {self._tcp_frame()})."
            )
            return

        self.get_logger().info(f"Switched command -> {cmd_txt} (frame_mode={self._mode()})")

    def toggle_mode(self):
        cur = self._mode()
        new_mode = "tcp" if cur == "camera" else "camera"
        self.set_parameters([Parameter("command_frame_mode", value=new_mode)])
        self.get_logger().info(f"Frame mode switched: {cur} -> {new_mode}")

        if self._tcp_twist_to_base_twist(np.zeros((6,), dtype=np.float64)) is None:
            self.get_logger().warn(
                f"Mode {new_mode} active, but TF not ready yet ({self._base_frame()} <- {self._tcp_frame()})."
            )
        if new_mode == "camera" and self._cam_twist_to_tcp_twist(np.zeros((6,), dtype=np.float64)) is None:
            self.get_logger().warn(
                f"Camera mode active, but TF not ready yet ({self._tcp_frame()} <- {self._camera_frame()})."
            )

    def publish_zero(self):
        self.pub.publish(Twist())


def run_menu(node: CameraRotationCalibrator):
    print("\nIBVS Camera-Rotation Calibrator (TF-based)")
    print("Menü:")
    print("  tx) Starte Translation +X")
    print("  ty) Starte Translation +Y")
    print("  tz) Starte Translation +Z")
    print("  rx) Starte Rotation +X")
    print("  ry) Starte Rotation +Y")
    print("  rz) Starte Rotation +Z")
    print("  m)  Umschalten Frame-Mode: camera <-> tcp")
    print("  0) Nullkommando senden (stop)")
    print("  q) Beenden")

    while rclpy.ok():
        try:
            choice = input("\nAuswahl [tx|ty|tz|rx|ry|rz|m|0|q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBeende ...")
            break

        if choice in ("tx", "ty", "tz", "rx", "ry", "rz"):
            node.set_command(choice)
        elif choice == "m":
            node.toggle_mode()
        elif choice == "0":
            node.set_command(None)
            node.publish_zero()
        elif choice == "q":
            break
        else:
            print("Ungültige Eingabe.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive camera-rotation calibration publisher using TF-based camera->tcp twist conversion."
        )
    )
    parser.add_argument(
        "--twist-topic",
        default="/cartesian_twist_passthrough_controller/cmd_vel",
        help="Twist topic for robot passthrough controller",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=30.0,
        help="Publish frequency in Hz",
    )
    parser.add_argument(
        "--linear-speed-m-s",
        type=float,
        default=0.01,
        help="Requested translational speed magnitude [m/s]",
    )
    parser.add_argument(
        "--angular-speed-rad-s",
        type=float,
        default=0.05,
        help="Requested camera-frame angular speed magnitude [rad/s]",
    )
    parser.add_argument(
        "--command-frame-mode",
        default="camera",
        choices=["camera", "tcp"],
        help="Interpret input commands in camera frame (TF-transform) or directly in tcp frame",
    )
    parser.add_argument(
        "--base-frame",
        default="base",
        help="Base frame used by twist passthrough controller interpretation",
    )
    parser.add_argument(
        "--tcp-frame",
        default="tool0",
        help="TCP frame used as target for twist command frame conversion",
    )
    parser.add_argument(
        "--camera-frame",
        default="camera_color_optical_frame",
        help="Camera frame used for desired camera-centric rotation input",
    )
    parser.add_argument(
        "--tf-lookup-timeout-sec",
        type=float,
        default=0.2,
        help="TF lookup timeout in seconds",
    )

    args = parser.parse_args()
    if args.publish_rate_hz <= 0.0:
        parser.error("--publish-rate-hz must be > 0")
    if args.linear_speed_m_s < 0.0:
        parser.error("--linear-speed-m-s must be >= 0")
    if args.angular_speed_rad_s < 0.0:
        parser.error("--angular-speed-rad-s must be >= 0")
    if args.tf_lookup_timeout_sec <= 0.0:
        parser.error("--tf-lookup-timeout-sec must be > 0")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = CameraRotationCalibrator(args)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_menu(node)
    finally:
        node.set_command(None)
        for _ in range(3):
            node.publish_zero()
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
