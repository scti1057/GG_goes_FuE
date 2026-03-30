#!/usr/bin/env python3
"""Interactive benchmark manager (step 1).

Current scope:
- Save start pose (base frame xyz + quaternion) from current TCP pose.
- Save goal pose (base frame xyz + quaternion) from current TCP pose.
- Drive to saved start/goal using translational and rotational cmd_vel control.
- Create benchmark session folder with stage subfolders.

Future benchmark logging/metrics will be added incrementally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String, UInt32


STAGE_KEYPOINT = "stage_1_keypoint_only"
STAGE_FILTER = "stage_2_filter"
STAGE_FILTER_LR = "stage_3_filter_local_rescue"

CONTROLLER_SPEED_PROFILES: dict[str, dict[str, Any]] = {
    "slow": {
        "label": "slow",
        "params": {
            "lambda_gain": 0.22,
            "max_linear_speed": 0.012,
            "max_angular_speed": 0.08,
            "smooth_cmd_enable": True,
            "smooth_cmd_use_median": True,
            "smooth_cmd_median_window": 3,
            "smooth_cmd_ema_alpha": 0.35,
            "smooth_cmd_max_linear_accel": 0.08,
            "smooth_cmd_max_angular_accel": 0.70,
        },
    },
    "mid": {
        "label": "mid",
        "params": {
            "lambda_gain": 0.28,
            "max_linear_speed": 0.018,
            "max_angular_speed": 0.12,
            "smooth_cmd_enable": True,
            "smooth_cmd_use_median": True,
            "smooth_cmd_median_window": 3,
            "smooth_cmd_ema_alpha": 0.45,
            "smooth_cmd_max_linear_accel": 0.12,
            "smooth_cmd_max_angular_accel": 1.00,
        },
    },
    "fast": {
        "label": "fast",
        "params": {
            "lambda_gain": 0.34,
            "max_linear_speed": 0.024,
            "max_angular_speed": 0.18,
            "smooth_cmd_enable": True,
            "smooth_cmd_use_median": False,
            "smooth_cmd_median_window": 1,
            "smooth_cmd_ema_alpha": 0.60,
            "smooth_cmd_max_linear_accel": 0.20,
            "smooth_cmd_max_angular_accel": 1.60,
        },
    },
    "fast_plus": {
        "label": "fast_plus",
        "params": {
            "lambda_gain": 0.40,
            "max_linear_speed": 0.030,
            "max_angular_speed": 0.24,
            "smooth_cmd_enable": True,
            "smooth_cmd_use_median": False,
            "smooth_cmd_median_window": 1,
            "smooth_cmd_ema_alpha": 0.72,
            "smooth_cmd_max_linear_accel": 0.30,
            "smooth_cmd_max_angular_accel": 2.20,
        },
    },
    "max": {
        "label": "max",
        "params": {
            "lambda_gain": 0.46,
            "max_linear_speed": 0.036,
            "max_angular_speed": 0.30,
            "smooth_cmd_enable": True,
            "smooth_cmd_use_median": False,
            "smooth_cmd_median_window": 1,
            "smooth_cmd_ema_alpha": 0.82,
            "smooth_cmd_max_linear_accel": 0.45,
            "smooth_cmd_max_angular_accel": 3.00,
        },
    },
}


def normalize_speed_profile_name(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in CONTROLLER_SPEED_PROFILES:
        return raw
    if raw == "1":
        return "slow"
    if raw == "2":
        return "mid"
    if raw == "3":
        return "fast"
    if raw == "4":
        return "fast_plus"
    if raw == "5":
        return "max"
    return "slow"


DEFAULTS: dict[str, Any] = {
    "pose_topic": "/tcp_pose_broadcaster/pose",
    "cmd_vel_topic": "/cartesian_twist_passthrough_controller/cmd_vel",
    "controller_speed_profile": "slow",
    "linear_kp": 1.2,
    "max_linear_speed": 0.03,
    "angular_kp": 1.4,
    "max_angular_speed": 0.45,
    "position_tolerance_m": 0.003,
    "orientation_tolerance_rad": 0.04,
    "goal_hold_sec": 0.35,
    "move_timeout_sec": 20.0,
    "pose_stale_timeout_sec": 1.0,
    "control_rate_hz": 30.0,
    "benchmark_repetitions": 5,
    "benchmark_timeout_sec": 20.0,
    "benchmark_pre_run_wait_sec": 5.0,
    "init_wait_timeout_sec": 20.0,
    "default_filter_type": "ekf",
    "detector_type": "xfeat",
    "detector_device": "cuda",
    "detector_top_k": 1024,
    "xfeat_repo_dir": "",
    "keypoints_topic": "/ibvs/keypoints",
    "reference_topic": "/ibvs/reference/keypoints",
    "matches_topic": "/ibvs/matches",
    "reference_init_duration_sec": 5.0,
    "reference_top_k": 300,
    "match_threshold": 0.85,
    "mutual_check": True,
    "prefilter_enabled": False,
    "prefilter_top_k": 100,
    "filter_q_noise": 2.0,
    "filter_r_noise": 1.1,
    "filter_gate_threshold": 20.0,
    "filter_z_depth": 0.25,
    "filter_predict_rate": 120.0,
    "filter_max_active_keypoints": 30,
    "filter_min_init_keypoints": 8,
    "filter_min_update_keypoints": 1,
    "filter_camera_velocity_topic": "/cartesian_twist_passthrough_controller/cmd_vel",
    "camera_node_name": "/camera/camera",
    "tracking_keep_aligned_depth": True,
    "goal_reached_topic": "/ibvs/control/goal_reached",
    "wx_debug_topic": "/ibvs/control/wx_debug",
    "local_rescue_mode_topic_node": "/descriptor_matcher_node",
    "keypoint_node_name": "/keypoint_node",
    "controller_node_name": "/ibvs_twist_controller_node",
    "filter_node_name": "/ibvs_filter_node",
    "keypoint_input_topic": "/camera/camera/color/image_raw/compressed",
    "keypoint_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth",
    "start_pose": None,
    "goal_pose": None,
}


@dataclass
class PoseTarget:
    xyz: np.ndarray
    quat_xyzw: Optional[np.ndarray]


@dataclass
class SavedTargets:
    start: Optional[PoseTarget] = None
    goal: Optional[PoseTarget] = None


@dataclass
class ManagedProcess:
    name: str
    cmd: list[str]
    log_path: Path
    process: Optional[subprocess.Popen] = None
    _log_handle: Optional[object] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> bool:
        if self.is_running():
            return False
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {self.name}\n"
            f"CMD: {' '.join(shlex.quote(x) for x in self.cmd)}\n\n"
        )
        self._log_handle.flush()
        self.process = subprocess.Popen(
            self.cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
        return True

    def stop(self, timeout_sec: float = 5.0) -> bool:
        if self.process is None:
            return False
        was_running = self.is_running()
        if was_running:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2.0)
        if self._log_handle is not None:
            self._log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] STOP {self.name}\n")
            self._log_handle.close()
            self._log_handle = None
        self.process = None
        return was_running


def run_cmd(cmd: list[str], timeout_sec: float = 8.0) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (cp.stdout or "") + (cp.stderr or "")
        return cp.returncode, out.strip()
    except subprocess.TimeoutExpired as exc:
        def _to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)

        out = _to_text(exc.stdout) + _to_text(exc.stderr)
        return 124, out.strip() if out.strip() else f"Timeout after {timeout_sec:.1f}s: {' '.join(cmd)}"


def vec3_to_str(v: np.ndarray) -> str:
    return f"[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]"


def quat_to_str(q: Optional[np.ndarray]) -> str:
    if q is None:
        return "not set"
    return f"[{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f}]"


def pose_to_str(p: Optional[PoseTarget]) -> str:
    if p is None:
        return "not set"
    return f"xyz={vec3_to_str(p.xyz)} quat={quat_to_str(p.quat_xyzw)}"


def pose_delta(start: Optional[PoseTarget], goal: Optional[PoseTarget]) -> tuple[Optional[float], Optional[float]]:
    if start is None or goal is None:
        return None, None
    pos_d = float(np.linalg.norm(goal.xyz - start.xyz))
    ori_d = None
    if start.quat_xyzw is not None and goal.quat_xyzw is not None:
        ori_d = float(np.linalg.norm(quat_error_rotvec(start.quat_xyzw, goal.quat_xyzw)))
    return pos_d, ori_d


def resolve_default_log_root() -> Path:
    in_container = Path("/home/ros_ws")
    if in_container.exists():
        return in_container / "logs" / "benchmark_csv"
    return Path("ros_ws/logs/benchmark_csv")


def parse_vec3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        arr = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
        return arr
    except Exception:
        return None


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def parse_quat4(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        arr = np.array(
            [float(value[0]), float(value[1]), float(value[2]), float(value[3])],
            dtype=np.float64,
        )
    except Exception:
        return None
    return normalize_quat_xyzw(arr)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_error_rotvec(current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    q_cur = normalize_quat_xyzw(current_q)
    q_tgt = normalize_quat_xyzw(target_q)
    q_err = quat_multiply(q_tgt, quat_conjugate(q_cur))
    q_err = normalize_quat_xyzw(q_err)
    if q_err[3] < 0.0:
        q_err = -q_err
    v = q_err[:3]
    s = float(np.linalg.norm(v))
    if s <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(s, float(q_err[3])))
    axis = v / s
    return axis * angle


def parse_pose_entry(value: Any) -> Optional[PoseTarget]:
    if not isinstance(value, dict):
        return None
    xyz = parse_vec3(value.get("xyz"))
    if xyz is None:
        return None
    quat = parse_quat4(value.get("quat_xyzw"))
    return PoseTarget(xyz=xyz, quat_xyzw=quat)


def parse_legacy_pose_entry(xyz_value: Any) -> Optional[PoseTarget]:
    xyz = parse_vec3(xyz_value)
    if xyz is None:
        return None
    return PoseTarget(xyz=xyz, quat_xyzw=None)


def pose_to_json_value(p: Optional[PoseTarget]) -> Any:
    if p is None:
        return None
    return {
        "xyz": p.xyz.tolist(),
        "quat_xyzw": None if p.quat_xyzw is None else p.quat_xyzw.tolist(),
    }


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception:
        return {}


def save_config_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def create_session_dirs(log_root: Path) -> Path:
    run_id = time.strftime("bench_%Y%m%d_%H%M%S")
    session_dir = log_root / run_id
    (session_dir / STAGE_KEYPOINT).mkdir(parents=True, exist_ok=True)
    (session_dir / STAGE_FILTER).mkdir(parents=True, exist_ok=True)
    (session_dir / STAGE_FILTER_LR).mkdir(parents=True, exist_ok=True)
    return session_dir


class BenchmarkManagerNode(Node):
    def __init__(self, args: argparse.Namespace, config_path: Path, log_root: Path):
        super().__init__("benchmark_manager")
        self.config_path = config_path
        self.log_root = log_root
        self.pose_topic = args.pose_topic
        self.cmd_vel_topic = args.cmd_vel_topic
        self.controller_speed_profile = normalize_speed_profile_name(args.controller_speed_profile)

        self.linear_kp = float(args.linear_kp)
        self.max_linear_speed = float(args.max_linear_speed)
        self.angular_kp = float(args.angular_kp)
        self.max_angular_speed = float(args.max_angular_speed)
        self.position_tolerance_m = float(args.position_tolerance_m)
        self.orientation_tolerance_rad = float(args.orientation_tolerance_rad)
        self.goal_hold_sec = float(args.goal_hold_sec)
        self.move_timeout_sec = float(args.move_timeout_sec)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        self.control_rate_hz = float(args.control_rate_hz)
        self.benchmark_repetitions = int(args.benchmark_repetitions)
        self.benchmark_timeout_sec = float(args.benchmark_timeout_sec)
        self.benchmark_pre_run_wait_sec = float(args.benchmark_pre_run_wait_sec)
        self.init_wait_timeout_sec = float(args.init_wait_timeout_sec)
        self.default_filter_type = str(args.default_filter_type)
        self.detector_type = str(args.detector_type)
        self.detector_device = str(args.detector_device)
        self.detector_top_k = int(args.detector_top_k)
        self.xfeat_repo_dir = str(args.xfeat_repo_dir)
        self.keypoints_topic = str(args.keypoints_topic)
        self.reference_topic = str(args.reference_topic)
        self.matches_topic = str(args.matches_topic)
        self.reference_init_duration_sec = float(args.reference_init_duration_sec)
        self.reference_top_k = int(args.reference_top_k)
        self.match_threshold = float(args.match_threshold)
        self.mutual_check = bool(args.mutual_check)
        self.prefilter_enabled = bool(args.prefilter_enabled)
        self.prefilter_top_k = int(args.prefilter_top_k)
        self.filter_q_noise = float(args.filter_q_noise)
        self.filter_r_noise = float(args.filter_r_noise)
        self.filter_gate_threshold = float(args.filter_gate_threshold)
        self.filter_z_depth = float(args.filter_z_depth)
        self.filter_predict_rate = float(args.filter_predict_rate)
        self.filter_max_active_keypoints = int(args.filter_max_active_keypoints)
        self.filter_min_init_keypoints = int(args.filter_min_init_keypoints)
        self.filter_min_update_keypoints = int(args.filter_min_update_keypoints)
        self.filter_camera_velocity_topic = str(args.filter_camera_velocity_topic)
        self.camera_node_name = str(args.camera_node_name)
        self.tracking_keep_aligned_depth = bool(args.tracking_keep_aligned_depth)
        self.goal_reached_topic = str(args.goal_reached_topic)
        self.wx_debug_topic = str(args.wx_debug_topic)
        self.controller_node_name = str(args.controller_node_name)
        self.local_rescue_node = str(args.local_rescue_mode_topic_node)
        self.keypoint_node_name = str(args.keypoint_node_name)
        self.filter_node_name = str(args.filter_node_name)
        self.keypoint_input_topic = str(args.keypoint_input_topic)
        self.keypoint_depth_topic = str(args.keypoint_depth_topic)

        self.current_xyz: Optional[np.ndarray] = None
        self.current_quat_xyzw: Optional[np.ndarray] = None
        self.last_pose_wall_sec: Optional[float] = None

        self.goal_reached_state: bool = False
        self.last_ibvs_rms_px: Optional[float] = None
        self.last_filter_update_status: str = ""
        self.local_rescue_attempts: int = 0
        self.local_rescue_success: int = 0
        self.local_rescue_reject: int = 0

        self._last_cmd_time_sec: Optional[float] = None
        self._last_cmd_lin: Optional[np.ndarray] = None
        self._last_cmd_ang: Optional[np.ndarray] = None
        self.last_cmd_lin_l2: float = 0.0
        self.last_cmd_ang_l2: float = 0.0
        self.last_cmd_lin_acc_l2: float = 0.0
        self.last_cmd_ang_acc_l2: float = 0.0

        self._last_base_time_sec: Optional[float] = None
        self._last_base_xyz: Optional[np.ndarray] = None
        self._last_base_quat: Optional[np.ndarray] = None
        self._last_base_lin_vel: Optional[np.ndarray] = None
        self._last_base_ang_vel: Optional[np.ndarray] = None
        self.last_base_lin_vel_l2: float = 0.0
        self.last_base_ang_vel_l2: float = 0.0
        self.last_base_lin_acc_l2: float = 0.0
        self.last_base_ang_acc_l2: float = 0.0

        self.run_active: bool = False
        self.run_index_active: int = 0
        self.run_level: str = ""
        self.run_stage: str = ""
        self.run_scenario: str = ""
        self.run_started_wall_sec: float = 0.0
        self.run_csv_handle: Optional[object] = None
        self.run_csv_writer: Optional[csv.DictWriter] = None
        self.run_filter_reject_count: int = 0
        self.run_local_rescue_attempts_start: int = 0
        self.run_local_rescue_success_start: int = 0
        self.run_local_rescue_reject_start: int = 0
        self.run_goal_reached: bool = False
        self.run_timed_out: bool = False

        self.saved = SavedTargets()

        self.sub_pose = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.on_pose,
            qos_profile_sensor_data,
        )
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.sub_cmd = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.on_cmd_vel,
            qos_profile_sensor_data,
        )
        self.sub_goal = self.create_subscription(
            Bool,
            self.goal_reached_topic,
            self.on_goal_reached,
            qos_profile_sensor_data,
        )
        self.sub_wx_debug = self.create_subscription(
            String,
            self.wx_debug_topic,
            self.on_wx_debug,
            qos_profile_sensor_data,
        )
        self.sub_filter_update_status = self.create_subscription(
            String,
            "/ibvs/filter/update_status",
            self.on_filter_update_status,
            qos_profile_sensor_data,
        )
        self.sub_lr_attempts = self.create_subscription(
            UInt32,
            "/ibvs/matching/local_rescue_attempts",
            self.on_local_rescue_attempts,
            qos_profile_sensor_data,
        )
        self.sub_lr_success = self.create_subscription(
            UInt32,
            "/ibvs/matching/local_rescue_success",
            self.on_local_rescue_success,
            qos_profile_sensor_data,
        )
        self.sub_lr_reject = self.create_subscription(
            UInt32,
            "/ibvs/matching/local_rescue_reject",
            self.on_local_rescue_reject,
            qos_profile_sensor_data,
        )
        self.run_log_timer = self.create_timer(1.0 / max(1.0, self.control_rate_hz), self._on_run_log_timer)

    def on_pose(self, msg: PoseStamped) -> None:
        now = time.time()
        self.current_xyz = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        self.current_quat_xyzw = normalize_quat_xyzw(
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
        self.last_pose_wall_sec = now

        if self._last_base_time_sec is not None and self._last_base_xyz is not None and self._last_base_quat is not None:
            dt = now - self._last_base_time_sec
            if dt > 1e-6:
                lin_vel = (self.current_xyz - self._last_base_xyz) / dt
                ang_vel = quat_error_rotvec(self._last_base_quat, self.current_quat_xyzw) / dt
                self.last_base_lin_vel_l2 = float(np.linalg.norm(lin_vel))
                self.last_base_ang_vel_l2 = float(np.linalg.norm(ang_vel))
                if self._last_base_lin_vel is not None:
                    self.last_base_lin_acc_l2 = float(np.linalg.norm((lin_vel - self._last_base_lin_vel) / dt))
                if self._last_base_ang_vel is not None:
                    self.last_base_ang_acc_l2 = float(np.linalg.norm((ang_vel - self._last_base_ang_vel) / dt))
                self._last_base_lin_vel = lin_vel
                self._last_base_ang_vel = ang_vel

        self._last_base_time_sec = now
        self._last_base_xyz = self.current_xyz.copy()
        self._last_base_quat = self.current_quat_xyzw.copy()

    def on_cmd_vel(self, msg: Twist) -> None:
        now = time.time()
        lin = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64)
        ang = np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64)
        self.last_cmd_lin_l2 = float(np.linalg.norm(lin))
        self.last_cmd_ang_l2 = float(np.linalg.norm(ang))

        if self._last_cmd_time_sec is not None and self._last_cmd_lin is not None and self._last_cmd_ang is not None:
            dt = now - self._last_cmd_time_sec
            if dt > 1e-6:
                self.last_cmd_lin_acc_l2 = float(np.linalg.norm((lin - self._last_cmd_lin) / dt))
                self.last_cmd_ang_acc_l2 = float(np.linalg.norm((ang - self._last_cmd_ang) / dt))
        self._last_cmd_time_sec = now
        self._last_cmd_lin = lin
        self._last_cmd_ang = ang

    def on_goal_reached(self, msg: Bool) -> None:
        self.goal_reached_state = bool(msg.data)
        if self.run_active and self.goal_reached_state:
            self.run_goal_reached = True

    def on_wx_debug(self, msg: String) -> None:
        try:
            obj = json.loads(msg.data)
            if isinstance(obj, dict) and "rms_px" in obj:
                v = obj["rms_px"]
                self.last_ibvs_rms_px = float(v) if v is not None else None
        except Exception:
            return

    def on_filter_update_status(self, msg: String) -> None:
        text = str(msg.data)
        self.last_filter_update_status = text
        if self.run_active and "REJECT" in text.upper():
            self.run_filter_reject_count += 1

    def on_local_rescue_attempts(self, msg: UInt32) -> None:
        self.local_rescue_attempts = int(msg.data)

    def on_local_rescue_success(self, msg: UInt32) -> None:
        self.local_rescue_success = int(msg.data)

    def on_local_rescue_reject(self, msg: UInt32) -> None:
        self.local_rescue_reject = int(msg.data)

    def start_run_csv(
        self,
        csv_path: Path,
        run_index: int,
        level_key: str,
        stage_key: str,
        scenario_key: str,
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_csv_handle = csv_path.open("w", newline="", encoding="utf-8")
        self.run_csv_writer = csv.DictWriter(
            self.run_csv_handle,
            fieldnames=[
                "t_rel_s",
                "run_index",
                "level",
                "stage",
                "scenario",
                "ibvs_rms_px",
                "ibvs_lin_vel_l2",
                "ibvs_ang_vel_l2",
                "ibvs_lin_acc_l2",
                "ibvs_ang_acc_l2",
                "base_lin_vel_l2",
                "base_ang_vel_l2",
                "base_lin_acc_l2",
                "base_ang_acc_l2",
                "filter_reject_count",
                "local_rescue_attempts_delta",
                "local_rescue_success_delta",
                "local_rescue_reject_delta",
                "run_timed_out",
                "goal_reached",
            ],
        )
        self.run_csv_writer.writeheader()
        self.run_csv_handle.flush()

        self.run_active = True
        self.run_started_wall_sec = time.time()
        self.run_index_active = int(run_index)
        self.run_level = level_key
        self.run_stage = stage_key
        self.run_scenario = scenario_key
        self.run_filter_reject_count = 0
        self.run_local_rescue_attempts_start = int(self.local_rescue_attempts)
        self.run_local_rescue_success_start = int(self.local_rescue_success)
        self.run_local_rescue_reject_start = int(self.local_rescue_reject)
        self.run_goal_reached = False
        self.run_timed_out = False

    def stop_run_csv(self) -> dict[str, Any]:
        self.run_active = False
        out = {
            "run_goal_reached": bool(self.run_goal_reached),
            "run_timed_out": bool(self.run_timed_out),
            "filter_reject_count": int(self.run_filter_reject_count),
            "local_rescue_attempts_delta": int(self.local_rescue_attempts - self.run_local_rescue_attempts_start),
            "local_rescue_success_delta": int(self.local_rescue_success - self.run_local_rescue_success_start),
            "local_rescue_reject_delta": int(self.local_rescue_reject - self.run_local_rescue_reject_start),
        }
        if self.run_csv_handle is not None:
            self.run_csv_handle.flush()
            self.run_csv_handle.close()
        self.run_csv_handle = None
        self.run_csv_writer = None
        return out

    def _on_run_log_timer(self) -> None:
        if not self.run_active or self.run_csv_writer is None:
            return
        now = time.time()
        row = {
            "t_rel_s": max(0.0, now - self.run_started_wall_sec),
            "run_index": self.run_index_active,
            "level": self.run_level,
            "stage": self.run_stage,
            "scenario": self.run_scenario,
            "ibvs_rms_px": self.last_ibvs_rms_px if self.last_ibvs_rms_px is not None else "",
            "ibvs_lin_vel_l2": self.last_cmd_lin_l2,
            "ibvs_ang_vel_l2": self.last_cmd_ang_l2,
            "ibvs_lin_acc_l2": self.last_cmd_lin_acc_l2,
            "ibvs_ang_acc_l2": self.last_cmd_ang_acc_l2,
            "base_lin_vel_l2": self.last_base_lin_vel_l2,
            "base_ang_vel_l2": self.last_base_ang_vel_l2,
            "base_lin_acc_l2": self.last_base_lin_acc_l2,
            "base_ang_acc_l2": self.last_base_ang_acc_l2,
            "filter_reject_count": self.run_filter_reject_count,
            "local_rescue_attempts_delta": int(self.local_rescue_attempts - self.run_local_rescue_attempts_start),
            "local_rescue_success_delta": int(self.local_rescue_success - self.run_local_rescue_success_start),
            "local_rescue_reject_delta": int(self.local_rescue_reject - self.run_local_rescue_reject_start),
            "run_timed_out": "true" if self.run_timed_out else "false",
            "goal_reached": "true" if self.goal_reached_state else "false",
        }
        self.run_csv_writer.writerow(row)
        if self.run_csv_handle is not None:
            self.run_csv_handle.flush()

    def wait_for_pose(self, timeout_sec: float = 3.0) -> bool:
        end = time.time() + timeout_sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_xyz is not None and self.current_quat_xyzw is not None:
                return True
        return self.current_xyz is not None and self.current_quat_xyzw is not None

    def set_start_from_current(self) -> bool:
        if self.current_xyz is None or self.current_quat_xyzw is None:
            return False
        self.saved.start = PoseTarget(
            xyz=self.current_xyz.copy(),
            quat_xyzw=self.current_quat_xyzw.copy(),
        )
        self.persist_config()
        return True

    def set_goal_from_current(self) -> bool:
        if self.current_xyz is None or self.current_quat_xyzw is None:
            return False
        self.saved.goal = PoseTarget(
            xyz=self.current_xyz.copy(),
            quat_xyzw=self.current_quat_xyzw.copy(),
        )
        self.persist_config()
        return True

    def config_dict(self) -> dict[str, Any]:
        return {
            "log_root": str(self.log_root),
            "pose_topic": self.pose_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "controller_speed_profile": self.controller_speed_profile,
            "linear_kp": float(self.linear_kp),
            "max_linear_speed": float(self.max_linear_speed),
            "angular_kp": float(self.angular_kp),
            "max_angular_speed": float(self.max_angular_speed),
            "position_tolerance_m": float(self.position_tolerance_m),
            "orientation_tolerance_rad": float(self.orientation_tolerance_rad),
            "goal_hold_sec": float(self.goal_hold_sec),
            "move_timeout_sec": float(self.move_timeout_sec),
            "pose_stale_timeout_sec": float(self.pose_stale_timeout_sec),
            "control_rate_hz": float(self.control_rate_hz),
            "benchmark_repetitions": int(self.benchmark_repetitions),
            "benchmark_timeout_sec": float(self.benchmark_timeout_sec),
            "benchmark_pre_run_wait_sec": float(self.benchmark_pre_run_wait_sec),
            "init_wait_timeout_sec": float(self.init_wait_timeout_sec),
            "default_filter_type": str(self.default_filter_type),
            "detector_type": self.detector_type,
            "detector_device": self.detector_device,
            "detector_top_k": int(self.detector_top_k),
            "xfeat_repo_dir": self.xfeat_repo_dir,
            "keypoints_topic": self.keypoints_topic,
            "reference_topic": self.reference_topic,
            "matches_topic": self.matches_topic,
            "reference_init_duration_sec": float(self.reference_init_duration_sec),
            "reference_top_k": int(self.reference_top_k),
            "match_threshold": float(self.match_threshold),
            "mutual_check": bool(self.mutual_check),
            "prefilter_enabled": bool(self.prefilter_enabled),
            "prefilter_top_k": int(self.prefilter_top_k),
            "filter_q_noise": float(self.filter_q_noise),
            "filter_r_noise": float(self.filter_r_noise),
            "filter_gate_threshold": float(self.filter_gate_threshold),
            "filter_z_depth": float(self.filter_z_depth),
            "filter_predict_rate": float(self.filter_predict_rate),
            "filter_max_active_keypoints": int(self.filter_max_active_keypoints),
            "filter_min_init_keypoints": int(self.filter_min_init_keypoints),
            "filter_min_update_keypoints": int(self.filter_min_update_keypoints),
            "filter_camera_velocity_topic": self.filter_camera_velocity_topic,
            "camera_node_name": self.camera_node_name,
            "tracking_keep_aligned_depth": bool(self.tracking_keep_aligned_depth),
            "goal_reached_topic": self.goal_reached_topic,
            "wx_debug_topic": self.wx_debug_topic,
            "local_rescue_mode_topic_node": self.local_rescue_node,
            "keypoint_node_name": self.keypoint_node_name,
            "controller_node_name": self.controller_node_name,
            "filter_node_name": self.filter_node_name,
            "keypoint_input_topic": self.keypoint_input_topic,
            "keypoint_depth_topic": self.keypoint_depth_topic,
            "start_pose": pose_to_json_value(self.saved.start),
            "goal_pose": pose_to_json_value(self.saved.goal),
        }

    def persist_config(self) -> None:
        save_config_atomic(self.config_path, self.config_dict())

    def publish_zero(self, repeats: int = 3) -> None:
        msg = Twist()
        for _ in range(max(1, repeats)):
            self.pub_cmd.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)

    def _pose_is_stale(self) -> bool:
        if self.last_pose_wall_sec is None:
            return True
        return (time.time() - self.last_pose_wall_sec) > self.pose_stale_timeout_sec

    def _compute_cmd(self, target: PoseTarget) -> tuple[Twist, float, float]:
        assert self.current_xyz is not None
        err = target.xyz - self.current_xyz
        dist_pos = float(np.linalg.norm(err))

        cmd = Twist()
        if dist_pos > 1e-12:
            v = self.linear_kp * err
            v_norm = float(np.linalg.norm(v))
            if v_norm > self.max_linear_speed and v_norm > 1e-12:
                v = v * (self.max_linear_speed / v_norm)

            cmd.linear.x = float(v[0])
            cmd.linear.y = float(v[1])
            cmd.linear.z = float(v[2])

        ori_err_norm = 0.0
        if target.quat_xyzw is not None and self.current_quat_xyzw is not None:
            rotvec = quat_error_rotvec(self.current_quat_xyzw, target.quat_xyzw)
            ori_err_norm = float(np.linalg.norm(rotvec))
            if ori_err_norm > 1e-12:
                w = self.angular_kp * rotvec
                w_norm = float(np.linalg.norm(w))
                if w_norm > self.max_angular_speed and w_norm > 1e-12:
                    w = w * (self.max_angular_speed / w_norm)
                cmd.angular.x = float(w[0])
                cmd.angular.y = float(w[1])
                cmd.angular.z = float(w[2])
        return cmd, dist_pos, ori_err_norm

    def move_to(self, name: str, target: PoseTarget) -> bool:
        if not self.wait_for_pose(timeout_sec=3.0):
            print("[move] no tcp pose available. abort.")
            return False

        print(f"[move] driving to {name}: {pose_to_str(target)}")
        start = time.time()
        hold_start: Optional[float] = None
        last_mode: Optional[str] = None
        dt = 1.0 / max(1.0, self.control_rate_hz)

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)

                if self.current_xyz is None or self._pose_is_stale():
                    if last_mode != "WAIT_POSE":
                        print("[move] waiting for fresh tcp pose...")
                        last_mode = "WAIT_POSE"
                    self.publish_zero(repeats=1)
                    if (time.time() - start) > self.move_timeout_sec:
                        print("[move] timeout while waiting for pose. abort.")
                        return False
                    time.sleep(dt)
                    continue

                cmd, dist_pos, dist_ori = self._compute_cmd(target)
                elapsed = time.time() - start
                if elapsed > self.move_timeout_sec:
                    print(
                        f"[move] timeout after {elapsed:.2f}s, pos_err={dist_pos:.4f} m, ori_err={dist_ori:.4f} rad"
                    )
                    return False

                in_pos = dist_pos <= self.position_tolerance_m
                in_ori = dist_ori <= self.orientation_tolerance_rad
                if in_pos and in_ori:
                    if hold_start is None:
                        hold_start = time.time()
                        if last_mode != "HOLD":
                            print(
                                "[move] within tolerance "
                                f"(pos={dist_pos:.4f} m, ori={dist_ori:.4f} rad). "
                                f"holding for {self.goal_hold_sec:.2f}s..."
                            )
                            last_mode = "HOLD"
                    self.publish_zero(repeats=1)
                    if (time.time() - hold_start) >= self.goal_hold_sec:
                        print(f"[move] goal reached in {elapsed:.2f}s")
                        return True
                else:
                    hold_start = None
                    if last_mode != "MOVE":
                        print("[move] moving...")
                        last_mode = "MOVE"
                    self.pub_cmd.publish(cmd)

                time.sleep(dt)
        finally:
            self.publish_zero(repeats=3)

        return False


class BenchmarkRuntime:
    def __init__(self, log_dir: Path, node: BenchmarkManagerNode):
        self.log_dir = log_dir
        self.node = node
        self.processes = self._build_processes()

    def _build_processes(self) -> dict[str, ManagedProcess]:
        keypoint_cmd = [
            "ros2",
            "run",
            "ibvs_perception",
            "keypoint",
            "--ros-args",
            "-p",
            f"detector_type:={self.node.detector_type}",
            "-p",
            f"device:={self.node.detector_device}",
            "-p",
            f"top_k:={self.node.detector_top_k}",
            "-p",
            "debug_mode:=false",
            "-p",
            f"input_topic:={self.node.keypoint_input_topic}",
            "-p",
            f"depth_topic:={self.node.keypoint_depth_topic}",
            "-p",
            "use_depth_roi:=true",
            "-p",
            f"keypoints_topic:={self.node.keypoints_topic}",
        ]
        if self.node.xfeat_repo_dir:
            keypoint_cmd += ["-p", f"xfeat_repo_dir:={self.node.xfeat_repo_dir}"]
        reference_cmd = [
            "ros2",
            "run",
            "ibvs_reference",
            "reference_manager",
            "--ros-args",
            "-p",
            f"keypoints_topic:={self.node.keypoints_topic}",
            "-p",
            f"image_topic:={self.node.keypoint_input_topic}",
            "-p",
            f"init_duration_sec:={self.node.reference_init_duration_sec}",
            "-p",
            f"ref_top_k:={self.node.reference_top_k}",
            "-p",
            "debug_mode:=false",
        ]
        matcher_cmd = [
            "ros2",
            "run",
            "ibvs_matching",
            "descriptor_matcher",
            "--ros-args",
            "-p",
            f"keypoints_topic:={self.node.keypoints_topic}",
            "-p",
            f"reference_topic:={self.node.reference_topic}",
            "-p",
            f"matches_topic:={self.node.matches_topic}",
            "-p",
            f"match_threshold:={self.node.match_threshold}",
            "-p",
            f"mutual_check:={'true' if self.node.mutual_check else 'false'}",
            "-p",
            f"prefilter_enabled:={'true' if self.node.prefilter_enabled else 'false'}",
            "-p",
            f"prefilter_top_k:={self.node.prefilter_top_k}",
        ]
        filter_cmd = self._make_filter_cmd(self.node.default_filter_type)
        controller_cmd = self._make_controller_cmd(level="c", feature_source="filtered")
        return {
            "keypoint": ManagedProcess("keypoint", keypoint_cmd, self.log_dir / "keypoint.log"),
            "reference_manager": ManagedProcess(
                "reference_manager", reference_cmd, self.log_dir / "reference_manager.log"
            ),
            "descriptor_matcher": ManagedProcess(
                "descriptor_matcher", matcher_cmd, self.log_dir / "descriptor_matcher.log"
            ),
            "filter_node": ManagedProcess("filter_node", filter_cmd, self.log_dir / "filter_node.log"),
            "ibvs_controller": ManagedProcess(
                "ibvs_controller", controller_cmd, self.log_dir / "ibvs_controller.log"
            ),
        }

    @staticmethod
    def _false_dofs_for_level(level: str) -> list[str]:
        lvl = str(level).strip().lower()
        if lvl == "a":
            return ["allow_wx", "allow_wy", "allow_wz"]
        if lvl == "b":
            return ["allow_wx", "allow_wy"]
        return []

    @staticmethod
    def _ros_param_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            txt = format(float(value), ".15g")
            if ("." not in txt) and ("e" not in txt) and ("E" not in txt):
                txt += ".0"
            return txt
        return str(value)

    def _controller_speed_params(self) -> dict[str, Any]:
        key = normalize_speed_profile_name(self.node.controller_speed_profile)
        return dict(CONTROLLER_SPEED_PROFILES[key]["params"])

    def _make_controller_cmd(self, level: str, feature_source: str) -> list[str]:
        src = str(feature_source).strip().lower()
        if src not in ("raw", "filtered"):
            src = "filtered"
        speed_params = self._controller_speed_params()
        cmd = [
            "ros2",
            "run",
            "ibvs_control",
            "ibvs_twist_controller",
            "--ros-args",
            "-p",
            "enable_motion:=false",
            "-p",
            "publish_wx_debug:=true",
            "-p",
            f"feature_source:={src}",
        ]
        for pname, pvalue in speed_params.items():
            cmd += ["-p", f"{pname}:={self._ros_param_text(pvalue)}"]
        for dof_param in self._false_dofs_for_level(level):
            cmd += ["-p", f"{dof_param}:=false"]
        return cmd

    def _make_filter_cmd(self, filter_type: str) -> list[str]:
        ft = str(filter_type).strip().lower()
        if ft not in ("ekf", "ukf", "eskf"):
            ft = "ekf"
        return [
            "ros2",
            "run",
            "ibvs_filter_cpp",
            "filter_node",
            "--ros-args",
            "-p",
            f"filter_type:={ft}",
            "-p",
            f"q_noise:={self.node.filter_q_noise}",
            "-p",
            f"r_noise:={self.node.filter_r_noise}",
            "-p",
            f"gate_threshold:={self.node.filter_gate_threshold}",
            "-p",
            f"z_depth:={self.node.filter_z_depth}",
            "-p",
            f"predict_rate:={self.node.filter_predict_rate}",
            "-p",
            f"camera_velocity_topic:={self.node.filter_camera_velocity_topic}",
            "-p",
            f"max_active_keypoints:={self.node.filter_max_active_keypoints}",
            "-p",
            f"min_init_keypoints:={self.node.filter_min_init_keypoints}",
            "-p",
            f"min_update_keypoints:={self.node.filter_min_update_keypoints}",
        ]

    def _start_proc(self, name: str) -> None:
        p = self.processes[name]
        started = p.start()
        if started:
            print(f"[proc] started {name} (pid={p.process.pid})")
        else:
            print(f"[proc] already running {name} (pid={p.process.pid})")

    def _stop_proc(self, name: str) -> None:
        p = self.processes[name]
        stopped = p.stop()
        if stopped:
            print(f"[proc] stopped {name}")
        else:
            print(f"[proc] already stopped {name}")

    def start_core(self) -> None:
        self._start_proc("keypoint")
        self._start_proc("reference_manager")

    def start_controller(self, level: str, feature_source: str) -> bool:
        self.processes["ibvs_controller"].cmd = self._make_controller_cmd(
            level=level,
            feature_source=feature_source,
        )
        self._start_proc("ibvs_controller")
        if not self.wait_for_node(self.node.controller_node_name, timeout_sec=8.0):
            print(f"[proc] controller node did not appear: {self.node.controller_node_name}")
            return False
        if not self.set_controller_enable_motion(False):
            print("[proc] warning: could not force enable_motion=false after start.")
        return True

    def stop_controller(self) -> None:
        if self.processes["ibvs_controller"].is_running():
            self.set_controller_enable_motion(False)
        self._stop_proc("ibvs_controller")

    def start_tracking(self) -> None:
        self._start_proc("descriptor_matcher")

    def stop_tracking(self) -> None:
        self._stop_proc("descriptor_matcher")

    def start_filter(self, filter_type: str) -> None:
        self.processes["filter_node"].cmd = self._make_filter_cmd(filter_type)
        self._start_proc("filter_node")

    def stop_filter(self) -> None:
        self._stop_proc("filter_node")

    def stop_all(self) -> None:
        self.stop_controller()
        self.stop_filter()
        self.stop_tracking()
        self._stop_proc("reference_manager")
        self._stop_proc("keypoint")

    def _set_param(self, node_name: str, param_name: str, value: str, timeout_sec: float = 6.0) -> bool:
        return self._set_param_retry(
            node_name=node_name,
            param_name=param_name,
            value=value,
            retries=4,
            timeout_sec=timeout_sec,
            retry_wait_sec=0.35,
        )

    def _set_param_retry(
        self,
        node_name: str,
        param_name: str,
        value: str,
        retries: int = 3,
        timeout_sec: float = 6.0,
        retry_wait_sec: float = 0.3,
    ) -> bool:
        retries = max(1, int(retries))
        for attempt in range(1, retries + 1):
            rc, out = run_cmd(
                ["ros2", "param", "set", node_name, param_name, value],
                timeout_sec=timeout_sec,
            )
            if rc == 0:
                return True
            short = out.splitlines()[-1] if out else "(no output)"
            print(
                f"[param] failed {node_name}.{param_name}={value} "
                f"({attempt}/{retries}): {short}"
            )
            if attempt < retries and retry_wait_sec > 0.0:
                time.sleep(retry_wait_sec)
        return False

    @staticmethod
    def _parse_ros_bool_param_get(output: str) -> Optional[bool]:
        out = str(output).strip().lower()
        m = re.search(r"boolean value is:\s*(true|false)", out)
        if m:
            return m.group(1) == "true"
        if re.search(r"\btrue\b", out):
            return True
        if re.search(r"\bfalse\b", out):
            return False
        return None

    def _set_node_param_bool(
        self,
        node_name: str,
        param_name: str,
        target_value: bool,
        retries: int = 3,
        timeout_sec: float = 6.0,
        retry_wait_sec: float = 0.4,
        skip_if_already_target: bool = False,
    ) -> bool:
        desired = "true" if target_value else "false"

        if skip_if_already_target:
            rc_get, out_get = run_cmd(
                ["ros2", "param", "get", node_name, param_name],
                timeout_sec=max(1.0, timeout_sec),
            )
            if rc_get == 0:
                current = self._parse_ros_bool_param_get(out_get)
                if current is not None and current == target_value:
                    print(f"[param] {node_name}.{param_name} already {desired}, skip set.")
                    return True

        for attempt in range(1, max(1, int(retries)) + 1):
            rc_set, out_set = run_cmd(
                ["ros2", "param", "set", node_name, param_name, desired],
                timeout_sec=timeout_sec,
            )
            if rc_set == 0:
                rc_get, out_get = run_cmd(
                    ["ros2", "param", "get", node_name, param_name],
                    timeout_sec=max(1.0, timeout_sec),
                )
                if rc_get == 0:
                    got = self._parse_ros_bool_param_get(out_get)
                    if got is None or got == target_value:
                        return True
                    print(
                        f"[param] verify mismatch {node_name}.{param_name}: got={got} "
                        f"expected={target_value} ({attempt}/{retries})"
                    )
                else:
                    short = out_get.splitlines()[-1] if out_get else "(no output)"
                    print(
                        f"[param] verify failed {node_name}.{param_name} "
                        f"({attempt}/{retries}): {short}"
                    )
            else:
                short = out_set.splitlines()[-1] if out_set else "(no output)"
                print(
                    f"[param] set failed {node_name}.{param_name}={desired} "
                    f"({attempt}/{retries}): {short}"
                )
            if attempt < retries and retry_wait_sec > 0.0:
                time.sleep(retry_wait_sec)
        return False

    def is_node_running(self, node_name: str) -> bool:
        rc, out = run_cmd(["ros2", "node", "list"], timeout_sec=4.0)
        if rc != 0:
            return False
        return str(node_name).strip() in set(x.strip() for x in out.splitlines() if x.strip())

    def wait_for_node(self, node_name: str, timeout_sec: float = 6.0) -> bool:
        deadline = time.time() + max(0.0, float(timeout_sec))
        while time.time() < deadline:
            if self.is_node_running(node_name):
                return True
            time.sleep(0.2)
        return self.is_node_running(node_name)

    def set_controller_enable_motion(self, enabled: bool) -> bool:
        return self._set_param_retry(
            self.node.controller_node_name,
            "enable_motion",
            "true" if enabled else "false",
            retries=6,
            timeout_sec=3.5,
            retry_wait_sec=0.35,
        )

    def set_local_rescue_mode(self, mode: str) -> bool:
        mode = str(mode).strip().lower()
        if mode not in ("off", "shadow", "active"):
            print(f"[local_rescue] invalid mode: {mode}")
            return False
        if not self.is_node_running(self.node.local_rescue_node):
            print(f"[local_rescue] matcher node not running ({self.node.local_rescue_node}), skip set.")
            return True
        return self._set_param(self.node.local_rescue_node, "local_rescue_mode", mode)

    def set_controller_level(self, level: str) -> bool:
        level = str(level).strip().lower()
        allow = {
            "allow_vx": True,
            "allow_vy": True,
            "allow_vz": True,
            "allow_wx": False,
            "allow_wy": False,
            "allow_wz": False,
        }
        if level == "b":
            allow["allow_wz"] = True
        elif level == "c":
            allow["allow_wx"] = True
            allow["allow_wy"] = True
            allow["allow_wz"] = True
        elif level != "a":
            print(f"[level] invalid level: {level}")
            return False

        ok = True
        for name, value in allow.items():
            ok &= self._set_param_retry(
                self.node.controller_node_name,
                name,
                "true" if value else "false",
                retries=4,
                timeout_sec=3.0,
                retry_wait_sec=0.25,
            )
        return ok

    def set_keypoint_use_depth_roi(self, enabled: bool) -> bool:
        if not self.is_node_running(self.node.keypoint_node_name):
            print(
                f"[keypoint] node not running ({self.node.keypoint_node_name}), "
                "skip use_depth_roi set."
            )
            return False
        ok = self._set_node_param_bool(
            self.node.keypoint_node_name,
            "use_depth_roi",
            bool(enabled),
            retries=3,
            timeout_sec=6.0,
            retry_wait_sec=0.35,
            skip_if_already_target=True,
        )
        if ok:
            time.sleep(0.2)
        return ok

    def set_camera_align_depth(self, enabled: bool) -> bool:
        if not self.is_node_running(self.node.camera_node_name):
            print(
                f"[camera] node not running ({self.node.camera_node_name}), "
                "skip align_depth set."
            )
            return False
        ok = self._set_node_param_bool(
            self.node.camera_node_name,
            "align_depth.enable",
            bool(enabled),
            retries=3,
            timeout_sec=6.0,
            retry_wait_sec=0.5,
            skip_if_already_target=True,
        )
        if ok:
            time.sleep(0.25)
        return ok

    def set_controller_feature_source(self, source: str) -> bool:
        src = str(source).strip().lower()
        if src not in ("raw", "filtered"):
            print(f"[controller] invalid feature_source: {src}")
            return False
        return self._set_param_retry(
            self.node.controller_node_name,
            "feature_source",
            src,
            retries=4,
            timeout_sec=3.0,
            retry_wait_sec=0.25,
        )

    def call_start_capture_service(self) -> bool:
        rc, out = run_cmd(
            ["ros2", "service", "call", "/ibvs/reference/start_capture", "std_srvs/srv/Trigger", "{}"],
            timeout_sec=12.0,
        )
        if rc != 0:
            short = out.splitlines()[-1] if out else "(no output)"
            print(f"[init] service call failed: {short}")
            return False
        ok = "success=True" in out
        print("[init] service response:", out.splitlines()[-1] if out else "(empty)")
        return ok

    def read_init_done(self) -> Optional[bool]:
        rc, out = run_cmd(
            [
                "ros2",
                "topic",
                "echo",
                "/ibvs/init_done",
                "--once",
                "--qos-durability",
                "transient_local",
                "--qos-reliability",
                "reliable",
                "--qos-history",
                "keep_last",
                "--qos-depth",
                "1",
            ],
            timeout_sec=6.0,
        )
        low = out.lower()
        if "data: true" in low:
            return True
        if "data: false" in low:
            return False
        if rc != 0:
            return None
        return None

    def wait_for_init_done(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            state = self.read_init_done()
            if state is True:
                return True
            time.sleep(0.5)
        return False


def print_menu() -> None:
    print("\nBenchmark Manager")
    print("  1) Set/Drive positions")
    print("  2) Manual benchmarking")
    print("  3) Automated benchmarking (later)")
    print("  4) Set controller speed")
    print("  5) Show status")
    print("  q) Quit")


def print_status(node: BenchmarkManagerNode, session_dir: Path) -> None:
    rclpy.spin_once(node, timeout_sec=0.05)
    cur_xyz = "not set" if node.current_xyz is None else vec3_to_str(node.current_xyz)
    cur_q = quat_to_str(node.current_quat_xyzw)
    print("\nStatus")
    print(f"- session_dir: {session_dir}")
    print(f"- config_path: {node.config_path}")
    print(f"- pose_topic:  {node.pose_topic}")
    print(f"- cmd_topic:   {node.cmd_vel_topic}")
    print(f"- current_xyz: {cur_xyz}")
    print(f"- current_q:   {cur_q}")
    print(f"- start:       {pose_to_str(node.saved.start)}")
    print(f"- goal:        {pose_to_str(node.saved.goal)}")
    pos_d, ori_d = pose_delta(node.saved.start, node.saved.goal)
    if pos_d is not None:
        ori_txt = f"{ori_d:.4f} rad" if ori_d is not None else "n/a"
        print(f"- start->goal: pos_delta={pos_d:.4f} m, ori_delta={ori_txt}")
    print(f"- runs_n:      {node.benchmark_repetitions}")
    print(f"- run_timeout: {node.benchmark_timeout_sec:.1f}s")
    print(f"- pre_wait:    {node.benchmark_pre_run_wait_sec:.1f}s")
    sp = normalize_speed_profile_name(node.controller_speed_profile)
    print(f"- ctrl_speed:  {sp}")


def controller_speed_menu(node: BenchmarkManagerNode) -> None:
    while True:
        current = normalize_speed_profile_name(node.controller_speed_profile)
        print("\nController speed profile")
        print(f"  current: {current}")
        print("  1) slow")
        print("  2) mid")
        print("  3) fast")
        print("  4) fast_plus")
        print("  5) max")
        print("  b) Back")
        choice = input("speed> ").strip().lower()
        if choice == "b":
            return
        target = {
            "1": "slow",
            "2": "mid",
            "3": "fast",
            "4": "fast_plus",
            "5": "max",
        }.get(choice)
        if target is None:
            print("unknown input.")
            continue
        node.controller_speed_profile = target
        node.persist_config()
        print(f"[speed] controller profile set to: {target} (saved)")


def positions_submenu(node: BenchmarkManagerNode, runtime: BenchmarkRuntime) -> None:
    while True:
        print("\nSet/Drive positions")
        print("  1) Set start pose from current pose")
        print("  2) Set goal pose from current pose")
        print("  3) Drive to start pose")
        print("  4) Drive to goal pose")
        print("  b) Back")
        pos_d, ori_d = pose_delta(node.saved.start, node.saved.goal)
        if pos_d is not None:
            if pos_d <= max(0.005, node.position_tolerance_m * 2.0):
                print(
                    f"  [warn] start/goal very close: pos_delta={pos_d:.4f} m "
                    f"(tolerance={node.position_tolerance_m:.4f} m)"
                )
            if ori_d is not None and ori_d <= max(0.05, node.orientation_tolerance_rad * 2.0):
                print(
                    f"  [warn] start/goal orientation very close: ori_delta={ori_d:.4f} rad "
                    f"(tolerance={node.orientation_tolerance_rad:.4f} rad)"
                )
        choice = input("positions> ").strip().lower()

        if choice == "1":
            if not node.wait_for_pose(timeout_sec=2.0):
                print("[set-start] no pose available.")
                continue
            ok = node.set_start_from_current()
            if ok:
                print(f"[set-start] saved start: {pose_to_str(node.saved.start)}")
            else:
                print("[set-start] failed (no pose).")
        elif choice == "2":
            if not node.wait_for_pose(timeout_sec=2.0):
                print("[set-goal] no pose available.")
                continue
            ok = node.set_goal_from_current()
            if ok:
                print(f"[set-goal] saved goal: {pose_to_str(node.saved.goal)}")
            else:
                print("[set-goal] failed (no pose).")
        elif choice == "3":
            if node.saved.start is None:
                print("[move-start] start is not set.")
                continue
            runtime.stop_controller()
            node.move_to("start", node.saved.start)
        elif choice == "4":
            if node.saved.goal is None:
                print("[move-goal] goal is not set.")
                continue
            runtime.stop_controller()
            node.move_to("goal", node.saved.goal)
        elif choice == "b":
            return
        else:
            print("unknown input.")


def _sleep_with_spin(node: BenchmarkManagerNode, duration_sec: float) -> None:
    end = time.time() + max(0.0, float(duration_sec))
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)


def execute_manual_benchmark(
    node: BenchmarkManagerNode,
    runtime: BenchmarkRuntime,
    session_dir: Path,
    level_choice: str,
    stage_choice: str,
    scenario_choice: str,
    filter_type: str,
    repetitions: int,
) -> None:
    stage_dir_name = {
        "1": "stage_1_keypoint_only",
        "2": "stage_2_filter",
        "3": "stage_3_filter_local_rescue",
    }[stage_choice]
    stage_short = {"1": "stage1", "2": "stage2", "3": "stage3"}[stage_choice]
    scenario_key = {
        "1": "scenario1_ideal_conditions",
        "2": "scenario2_with_occlusion",
        "3": "scenario3_with_light_interference",
        "4": "scenario4_with_occlusion_and_light_interference",
    }[scenario_choice]
    level_key = {"a": "level_a", "b": "level_b", "c": "level_c"}[level_choice]

    if node.saved.start is None or node.saved.goal is None:
        print("[manual] start/goal pose missing. Set both first in menu 1.")
        return

    print("\n[manual] preparing benchmark run...")
    runtime.stop_controller()
    runtime.set_local_rescue_mode("off")
    runtime.stop_filter()
    runtime.stop_tracking()
    print("[manual] camera align_depth untouched (using driver default).")

    if not node.move_to("goal_for_initialization", node.saved.goal):
        print("[manual] could not reach goal pose for initialization. abort.")
        return

    if not runtime.set_keypoint_use_depth_roi(True):
        print("[manual] could not set keypoint use_depth_roi=true. abort.")
        return
    if not runtime.call_start_capture_service():
        print("[manual] initialization service failed. abort.")
        return
    print("[manual] waiting for init_done=true ...")
    if not runtime.wait_for_init_done(node.init_wait_timeout_sec):
        print("[manual] init_done timeout. abort.")
        return

    runtime.start_tracking()
    if not runtime.set_keypoint_use_depth_roi(False):
        print("[manual] warning: could not set keypoint use_depth_roi=false for tracking.")
    if not node.move_to("start_before_runs", node.saved.start):
        print("[manual] could not reach start pose before runs. abort.")
        return

    stage_dir = session_dir / stage_dir_name
    for run_idx in range(1, max(1, repetitions) + 1):
        print(f"\n[manual] run {run_idx}/{repetitions}")
        runtime.stop_controller()
        node.goal_reached_state = False
        node.run_goal_reached = False
        node.run_timed_out = False

        if not node.move_to("start_reset", node.saved.start):
            print("[manual] could not return to start pose. abort remaining runs.")
            break

        if stage_choice in ("2", "3"):
            runtime.stop_filter()
            runtime.start_filter(filter_type)
            _sleep_with_spin(node, 1.0)
        else:
            runtime.stop_filter()

        if stage_choice == "3":
            runtime.set_local_rescue_mode("active")
        else:
            runtime.set_local_rescue_mode("off")

        if node.benchmark_pre_run_wait_sec > 0.0:
            print(f"[manual] settling {node.benchmark_pre_run_wait_sec:.1f}s before run ...")
            _sleep_with_spin(node, node.benchmark_pre_run_wait_sec)

        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_name = f"{level_key}_{stage_short}_{scenario_key}_run{run_idx:02d}_{ts}.csv"
        csv_path = stage_dir / csv_name
        node.start_run_csv(
            csv_path=csv_path,
            run_index=run_idx,
            level_key=level_key,
            stage_key=stage_short,
            scenario_key=scenario_key,
        )

        feature_source = "raw" if stage_choice == "1" else "filtered"
        if not runtime.start_controller(level=level_choice, feature_source=feature_source):
            print("[manual] controller start failed. abort remaining runs.")
            node.run_goal_reached = False
            node.run_timed_out = True
            summary = node.stop_run_csv()
            print(
                "[manual] run done "
                f"goal_reached={summary['run_goal_reached']} "
                f"timed_out={summary['run_timed_out']} "
                f"filter_rejects={summary['filter_reject_count']} "
                f"lr_attempts={summary['local_rescue_attempts_delta']} "
                f"lr_success={summary['local_rescue_success_delta']} "
                f"lr_reject={summary['local_rescue_reject_delta']} "
                f"csv={csv_path}"
            )
            break
        _sleep_with_spin(node, 0.8)
        if not runtime.set_controller_enable_motion(True):
            print("[manual] failed to enable controller motion. abort remaining runs.")
            node.run_goal_reached = False
            node.run_timed_out = True
            runtime.stop_controller()
            summary = node.stop_run_csv()
            print(
                "[manual] run done "
                f"goal_reached={summary['run_goal_reached']} "
                f"timed_out={summary['run_timed_out']} "
                f"filter_rejects={summary['filter_reject_count']} "
                f"lr_attempts={summary['local_rescue_attempts_delta']} "
                f"lr_success={summary['local_rescue_success_delta']} "
                f"lr_reject={summary['local_rescue_reject_delta']} "
                f"csv={csv_path}"
            )
            break
        start_wait = time.time()
        # Manual light cues for the operator:
        # - scenario 3: "light on" at +4s
        # - scenario 4: "light on" at +4s, "lights off" +6s later
        light_on_deadline: Optional[float] = None
        lights_off_deadline: Optional[float] = None
        light_on_done = False
        lights_off_done = False
        if scenario_choice in ("3", "4"):
            light_on_deadline = start_wait + 3.0
        if scenario_choice == "4":
            lights_off_deadline = start_wait + 5.0

        reached = False
        while rclpy.ok() and (time.time() - start_wait) <= node.benchmark_timeout_sec:
            now = time.time()
            if light_on_deadline is not None and (not light_on_done) and now >= light_on_deadline:
                print("light on")
                light_on_done = True
            if lights_off_deadline is not None and (not lights_off_done) and now >= lights_off_deadline:
                print("lights off")
                lights_off_done = True
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.goal_reached_state:
                reached = True
                break
            time.sleep(0.05)

        node.run_goal_reached = reached
        node.run_timed_out = not reached
        runtime.set_controller_enable_motion(False)
        runtime.stop_controller()
        summary = node.stop_run_csv()
        print(
            "[manual] run done "
            f"goal_reached={summary['run_goal_reached']} "
            f"timed_out={summary['run_timed_out']} "
            f"filter_rejects={summary['filter_reject_count']} "
            f"lr_attempts={summary['local_rescue_attempts_delta']} "
            f"lr_success={summary['local_rescue_success_delta']} "
            f"lr_reject={summary['local_rescue_reject_delta']} "
            f"csv={csv_path}"
        )

        runtime.set_local_rescue_mode("off")
        if stage_choice in ("2", "3"):
            runtime.stop_filter()

        if not node.move_to("start_after_run", node.saved.start):
            print("[manual] could not return to start after run. abort remaining runs.")
            break

    runtime.stop_controller()
    runtime.set_local_rescue_mode("off")
    runtime.stop_filter()
    print("[manual] benchmark combination finished.")


def manual_benchmark_menu(node: BenchmarkManagerNode, runtime: BenchmarkRuntime, session_dir: Path) -> None:
    levels: dict[str, tuple[str, str]] = {
        "a": ("level_a", "xyz translation, rotation freeze"),
        "b": ("level_b", "xyz translation + rotation z (x/y freeze)"),
        "c": ("level_c", "xyz translation + xyz rotation (all free)"),
    }
    stages: dict[str, tuple[str, str, str]] = {
        "1": ("stage_1_keypoint_only", "stage1", "keypoints"),
        "2": ("stage_2_filter", "stage2", "keypoints+filter"),
        "3": ("stage_3_filter_local_rescue", "stage3", "keypoints+filter+local_rescue"),
    }
    scenarios: dict[str, tuple[str, str]] = {
        "1": ("scenario1_ideal_conditions", "ideal conditions"),
        "2": ("scenario2_with_occlusion", "with occlusion"),
        "3": ("scenario3_with_light_interference", "with light interference"),
        "4": ("scenario4_with_occlusion_and_light_interference", "with occlusion and light interference"),
    }

    while True:
        print("\nManual benchmarking: select level")
        print("  a) Level a: xyz translation, rotation freeze")
        print("  b) Level b: xyz translation + rotation z")
        print("  c) Level c: xyz translation + xyz rotation")
        print("  x) Back")
        level_choice = input("manual/level> ").strip().lower()
        if level_choice == "x":
            return
        if level_choice not in levels:
            print("unknown input.")
            continue
        level_key, level_desc = levels[level_choice]

        while True:
            print("\nManual benchmarking: select stage")
            print("  1) Stage 1: keypoints")
            print("  2) Stage 2: keypoints + filter")
            print("  3) Stage 3: keypoints + filter + local rescue")
            print("  x) Back")
            stage_choice = input("manual/stage> ").strip().lower()
            if stage_choice == "x":
                break
            if stage_choice not in stages:
                print("unknown input.")
                continue
            stage_dir_name, stage_short, stage_desc = stages[stage_choice]

            while True:
                print("\nManual benchmarking: select scenario")
                print("  1) ideal conditions")
                print("  2) with occlusion")
                print("  3) with light interference")
                print("  4) with occlusion and light interference")
                print("  x) Back")
                scenario_choice = input("manual/scenario> ").strip().lower()
                if scenario_choice == "x":
                    break
                if scenario_choice not in scenarios:
                    print("unknown input.")
                    continue

                scenario_key, scenario_desc = scenarios[scenario_choice]
                stage_path = session_dir / stage_dir_name
                ts = time.strftime("%Y%m%d_%H%M%S")
                csv_name = f"{level_key}_{stage_short}_{scenario_key}_{ts}.csv"
                csv_path = stage_path / csv_name
                filter_type = node.default_filter_type
                if stage_choice in ("2", "3"):
                    print("\nFilter type for this benchmark:")
                    print("  1) ekf")
                    print("  2) ukf")
                    print("  3) eskf")
                    ft_choice = input("manual/filter> ").strip().lower()
                    filter_type = {"1": "ekf", "2": "ukf", "3": "eskf"}.get(ft_choice, node.default_filter_type)
                raw_n = input(
                    f"Repetitions n [default {node.benchmark_repetitions}]: "
                ).strip()
                if raw_n == "":
                    n_runs = node.benchmark_repetitions
                elif raw_n.isdigit():
                    n_runs = max(1, int(raw_n))
                else:
                    print("[manual] invalid n, using default.")
                    n_runs = node.benchmark_repetitions

                print("\nManual benchmark selection")
                print(f"- level:       {level_desc} ({level_key})")
                print(f"- stage:       {stage_desc} ({stage_short})")
                print(f"- scenario:    {scenario_desc} ({scenario_key})")
                print(f"- csv target:  {csv_path}")
                print(f"- filter:      {filter_type if stage_choice in ('2','3') else 'off'}")
                print(f"- runs (n):    {n_runs}")
                confirm = input("Start benchmark now? (y/N): ").strip().lower()
                if confirm != "y":
                    print("[manual] canceled.")
                    continue
                execute_manual_benchmark(
                    node=node,
                    runtime=runtime,
                    session_dir=session_dir,
                    level_choice=level_choice,
                    stage_choice=stage_choice,
                    scenario_choice=scenario_choice,
                    filter_type=filter_type,
                    repetitions=n_runs,
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive benchmark manager (step 1).")
    p.add_argument("--config-path", type=Path, default=None)
    p.add_argument("--pose-topic", default=None)
    p.add_argument("--cmd-vel-topic", default=None)
    p.add_argument("--log-root", type=Path, default=None)
    p.add_argument("--controller-speed-profile", default=None)
    p.add_argument("--linear-kp", type=float, default=None)
    p.add_argument("--max-linear-speed", type=float, default=None)
    p.add_argument("--angular-kp", type=float, default=None)
    p.add_argument("--max-angular-speed", type=float, default=None)
    p.add_argument("--position-tolerance-m", type=float, default=None)
    p.add_argument("--orientation-tolerance-rad", type=float, default=None)
    p.add_argument("--goal-hold-sec", type=float, default=None)
    p.add_argument("--move-timeout-sec", type=float, default=None)
    p.add_argument("--pose-stale-timeout-sec", type=float, default=None)
    p.add_argument("--control-rate-hz", type=float, default=None)
    p.add_argument("--benchmark-repetitions", type=int, default=None)
    p.add_argument("--benchmark-timeout-sec", type=float, default=None)
    p.add_argument("--benchmark-pre-run-wait-sec", type=float, default=None)
    p.add_argument("--init-wait-timeout-sec", type=float, default=None)
    p.add_argument("--default-filter-type", default=None)
    p.add_argument("--detector-type", default=None)
    p.add_argument("--detector-device", default=None)
    p.add_argument("--detector-top-k", type=int, default=None)
    p.add_argument("--xfeat-repo-dir", default=None)
    p.add_argument("--keypoints-topic", default=None)
    p.add_argument("--reference-topic", default=None)
    p.add_argument("--matches-topic", default=None)
    p.add_argument("--reference-init-duration-sec", type=float, default=None)
    p.add_argument("--reference-top-k", type=int, default=None)
    p.add_argument("--match-threshold", type=float, default=None)
    p.add_argument("--mutual-check", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefilter-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefilter-top-k", type=int, default=None)
    p.add_argument("--filter-q-noise", type=float, default=None)
    p.add_argument("--filter-r-noise", type=float, default=None)
    p.add_argument("--filter-gate-threshold", type=float, default=None)
    p.add_argument("--filter-z-depth", type=float, default=None)
    p.add_argument("--filter-predict-rate", type=float, default=None)
    p.add_argument("--filter-max-active-keypoints", type=int, default=None)
    p.add_argument("--filter-min-init-keypoints", type=int, default=None)
    p.add_argument("--filter-min-update-keypoints", type=int, default=None)
    p.add_argument("--filter-camera-velocity-topic", default=None)
    p.add_argument("--camera-node-name", default=None)
    p.add_argument("--tracking-keep-aligned-depth", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--goal-reached-topic", default=None)
    p.add_argument("--wx-debug-topic", default=None)
    p.add_argument("--local-rescue-mode-topic-node", default=None)
    p.add_argument("--keypoint-node-name", default=None)
    p.add_argument("--controller-node-name", default=None)
    p.add_argument("--filter-node-name", default=None)
    p.add_argument("--keypoint-input-topic", default=None)
    p.add_argument("--keypoint-depth-topic", default=None)
    return p.parse_args()


def resolve_effective_settings(cli: argparse.Namespace) -> tuple[argparse.Namespace, Path, Path]:
    default_log_root = resolve_default_log_root()
    log_root_for_config = Path(cli.log_root) if cli.log_root is not None else default_log_root
    config_path = (
        Path(cli.config_path)
        if cli.config_path is not None
        else (log_root_for_config / "benchmark_manager_config.json")
    )

    cfg = load_config(config_path)

    merged: dict[str, Any] = dict(DEFAULTS)
    if isinstance(cfg.get("log_root"), str) and cfg.get("log_root"):
        merged["log_root"] = cfg["log_root"]
    else:
        merged["log_root"] = str(default_log_root)

    for key in DEFAULTS.keys():
        if key in cfg:
            merged[key] = cfg[key]

    cli_overrides = {
        "pose_topic": cli.pose_topic,
        "cmd_vel_topic": cli.cmd_vel_topic,
        "log_root": str(cli.log_root) if cli.log_root is not None else None,
        "controller_speed_profile": cli.controller_speed_profile,
        "linear_kp": cli.linear_kp,
        "max_linear_speed": cli.max_linear_speed,
        "angular_kp": cli.angular_kp,
        "max_angular_speed": cli.max_angular_speed,
        "position_tolerance_m": cli.position_tolerance_m,
        "orientation_tolerance_rad": cli.orientation_tolerance_rad,
        "goal_hold_sec": cli.goal_hold_sec,
        "move_timeout_sec": cli.move_timeout_sec,
        "pose_stale_timeout_sec": cli.pose_stale_timeout_sec,
        "control_rate_hz": cli.control_rate_hz,
        "benchmark_repetitions": cli.benchmark_repetitions,
        "benchmark_timeout_sec": cli.benchmark_timeout_sec,
        "benchmark_pre_run_wait_sec": cli.benchmark_pre_run_wait_sec,
        "init_wait_timeout_sec": cli.init_wait_timeout_sec,
        "default_filter_type": cli.default_filter_type,
        "detector_type": cli.detector_type,
        "detector_device": cli.detector_device,
        "detector_top_k": cli.detector_top_k,
        "xfeat_repo_dir": cli.xfeat_repo_dir,
        "keypoints_topic": cli.keypoints_topic,
        "reference_topic": cli.reference_topic,
        "matches_topic": cli.matches_topic,
        "reference_init_duration_sec": cli.reference_init_duration_sec,
        "reference_top_k": cli.reference_top_k,
        "match_threshold": cli.match_threshold,
        "mutual_check": cli.mutual_check,
        "prefilter_enabled": cli.prefilter_enabled,
        "prefilter_top_k": cli.prefilter_top_k,
        "filter_q_noise": cli.filter_q_noise,
        "filter_r_noise": cli.filter_r_noise,
        "filter_gate_threshold": cli.filter_gate_threshold,
        "filter_z_depth": cli.filter_z_depth,
        "filter_predict_rate": cli.filter_predict_rate,
        "filter_max_active_keypoints": cli.filter_max_active_keypoints,
        "filter_min_init_keypoints": cli.filter_min_init_keypoints,
        "filter_min_update_keypoints": cli.filter_min_update_keypoints,
        "filter_camera_velocity_topic": cli.filter_camera_velocity_topic,
        "camera_node_name": cli.camera_node_name,
        "tracking_keep_aligned_depth": cli.tracking_keep_aligned_depth,
        "goal_reached_topic": cli.goal_reached_topic,
        "wx_debug_topic": cli.wx_debug_topic,
        "local_rescue_mode_topic_node": cli.local_rescue_mode_topic_node,
        "keypoint_node_name": cli.keypoint_node_name,
        "controller_node_name": cli.controller_node_name,
        "filter_node_name": cli.filter_node_name,
        "keypoint_input_topic": cli.keypoint_input_topic,
        "keypoint_depth_topic": cli.keypoint_depth_topic,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value

    # Benchmark default policy: use CUDA unless explicitly overridden via CLI.
    # This also upgrades old persisted configs that still carry detector_device=cpu.
    if cli.detector_device is None:
        merged["detector_device"] = "cuda"

    effective = argparse.Namespace(
        pose_topic=str(merged["pose_topic"]),
        cmd_vel_topic=str(merged["cmd_vel_topic"]),
        log_root=Path(str(merged["log_root"])),
        controller_speed_profile=normalize_speed_profile_name(merged.get("controller_speed_profile", "slow")),
        linear_kp=float(merged["linear_kp"]),
        max_linear_speed=float(merged["max_linear_speed"]),
        angular_kp=float(merged["angular_kp"]),
        max_angular_speed=float(merged["max_angular_speed"]),
        position_tolerance_m=float(merged["position_tolerance_m"]),
        orientation_tolerance_rad=float(merged["orientation_tolerance_rad"]),
        goal_hold_sec=float(merged["goal_hold_sec"]),
        move_timeout_sec=float(merged["move_timeout_sec"]),
        pose_stale_timeout_sec=float(merged["pose_stale_timeout_sec"]),
        control_rate_hz=float(merged["control_rate_hz"]),
        benchmark_repetitions=int(merged["benchmark_repetitions"]),
        benchmark_timeout_sec=float(merged["benchmark_timeout_sec"]),
        benchmark_pre_run_wait_sec=float(merged["benchmark_pre_run_wait_sec"]),
        init_wait_timeout_sec=float(merged["init_wait_timeout_sec"]),
        default_filter_type=str(merged["default_filter_type"]).lower(),
        detector_type=str(merged["detector_type"]),
        detector_device=str(merged["detector_device"]),
        detector_top_k=int(merged["detector_top_k"]),
        xfeat_repo_dir=str(merged["xfeat_repo_dir"]),
        keypoints_topic=str(merged["keypoints_topic"]),
        reference_topic=str(merged["reference_topic"]),
        matches_topic=str(merged["matches_topic"]),
        reference_init_duration_sec=float(merged["reference_init_duration_sec"]),
        reference_top_k=int(merged["reference_top_k"]),
        match_threshold=float(merged["match_threshold"]),
        mutual_check=bool(merged["mutual_check"]),
        prefilter_enabled=bool(merged["prefilter_enabled"]),
        prefilter_top_k=int(merged["prefilter_top_k"]),
        filter_q_noise=float(merged["filter_q_noise"]),
        filter_r_noise=float(merged["filter_r_noise"]),
        filter_gate_threshold=float(merged["filter_gate_threshold"]),
        filter_z_depth=float(merged["filter_z_depth"]),
        filter_predict_rate=float(merged["filter_predict_rate"]),
        filter_max_active_keypoints=int(merged["filter_max_active_keypoints"]),
        filter_min_init_keypoints=int(merged["filter_min_init_keypoints"]),
        filter_min_update_keypoints=int(merged["filter_min_update_keypoints"]),
        filter_camera_velocity_topic=str(merged["filter_camera_velocity_topic"]),
        camera_node_name=str(merged["camera_node_name"]),
        tracking_keep_aligned_depth=bool(merged["tracking_keep_aligned_depth"]),
        goal_reached_topic=str(merged["goal_reached_topic"]),
        wx_debug_topic=str(merged["wx_debug_topic"]),
        local_rescue_mode_topic_node=str(merged["local_rescue_mode_topic_node"]),
        keypoint_node_name=str(merged["keypoint_node_name"]),
        controller_node_name=str(merged["controller_node_name"]),
        filter_node_name=str(merged["filter_node_name"]),
        keypoint_input_topic=str(merged["keypoint_input_topic"]),
        keypoint_depth_topic=str(merged["keypoint_depth_topic"]),
        start_pose=merged.get("start_pose"),
        goal_pose=merged.get("goal_pose"),
    )

    start_pose = parse_pose_entry(effective.start_pose)
    goal_pose = parse_pose_entry(effective.goal_pose)
    if start_pose is None and "start_xyz" in merged:
        start_pose = parse_legacy_pose_entry(merged.get("start_xyz"))
    if goal_pose is None and "goal_xyz" in merged:
        goal_pose = parse_legacy_pose_entry(merged.get("goal_xyz"))
    effective.start_pose = start_pose
    effective.goal_pose = goal_pose

    return effective, effective.log_root, config_path


def main() -> int:
    if subprocess.run(["which", "ros2"], capture_output=True, text=True).returncode != 0:
        print("error: 'ros2' not found. source ROS first.")
        return 2

    cli_args = parse_args()
    args, log_root, config_path = resolve_effective_settings(cli_args)
    session_dir = create_session_dirs(log_root)
    proc_log_dir = session_dir / "process_logs"

    rclpy.init()
    node = BenchmarkManagerNode(args, config_path=config_path, log_root=log_root)
    node.saved.start = args.start_pose
    node.saved.goal = args.goal_pose
    runtime = BenchmarkRuntime(log_dir=proc_log_dir, node=node)
    node.persist_config()

    print(f"[session] created: {session_dir}")
    print(f"[session] stages: {STAGE_KEYPOINT}, {STAGE_FILTER}, {STAGE_FILTER_LR}")
    print(f"[config] path: {config_path}")
    print(f"[proc-logs] path: {proc_log_dir}")
    print(f"[startup] detector: type={node.detector_type} device={node.detector_device} top_k={node.detector_top_k}")
    print("[startup] starting core (keypoint + reference_manager) without debug mode...")
    runtime.start_core()
    if not node.wait_for_pose(timeout_sec=2.0):
        print(f"[warn] no pose on {node.pose_topic} yet. set/start/goal will wait for pose.")

    try:
        while rclpy.ok():
            print_menu()
            choice = input("> ").strip().lower()

            if choice == "1":
                positions_submenu(node, runtime)
            elif choice == "2":
                manual_benchmark_menu(node=node, runtime=runtime, session_dir=session_dir)
            elif choice == "3":
                print("[auto] automated benchmarking menu is not implemented yet.")
            elif choice == "4":
                controller_speed_menu(node)
            elif choice == "5":
                print_status(node, session_dir)
            elif choice == "q":
                break
            else:
                print("unknown input.")
    except KeyboardInterrupt:
        pass
    finally:
        node.persist_config()
        runtime.stop_all()
        node.publish_zero(repeats=3)
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
