#!/usr/bin/env python3
"""Compact IBVS debug logger (JSONL).

Writes downsampled, aggregated snapshots for quick jitter analysis.
Each line is one JSON object (JSONL) to keep memory usage low.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from ibvs_msgs.msg import Keypoints, Matches
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32, String, UInt32


@dataclass
class MatchStats:
    n: int = 0
    depth_valid: int = 0
    depth_ratio: float = 0.0
    depth_median_m: float = math.nan
    recv_sec: float = -1.0


@dataclass
class KeypointStats:
    n: int = 0
    depth_valid: int = 0
    depth_ratio: float = 0.0
    recv_sec: float = -1.0


@dataclass
class TwistStats:
    lx: float = 0.0
    ly: float = 0.0
    lz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    recv_sec: float = -1.0


class IbvsDebugJsonLogger(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ibvs_debug_json_logger")

        self.args = args
        self.start_wall_sec = time.time()
        self.start_mono_sec = time.monotonic()
        self.start_ros_sec = self.now_sec()
        self.sample_count = 0

        self.raw = MatchStats()
        self.filtered = MatchStats()
        self.kp = KeypointStats()
        self.cmd = TwistStats()
        self.tcp = TwistStats()

        self.goal_reached = False
        self.filter_uncertainty = math.nan
        self.filter_status = ""
        self.update_status = ""
        self.active_count = 0
        self.ctrl_wx_debug: Optional[dict] = None
        self.ctrl_wx_debug_recv_sec = -1.0

        self.prev_wx_sign = 0
        self.wx_sign_flips = 0
        self.wx_near_zero_samples = 0

        out_path = Path(args.out_jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.fh = out_path.open("w", encoding="utf-8")

        self.write_event(
            {
                "type": "meta",
                "utc_start": datetime.now(timezone.utc).isoformat(),
                "sample_hz": args.sample_hz,
                "duration_sec": args.duration_sec,
                "topics": {
                    "raw_matches": args.raw_matches_topic,
                    "filtered_matches": args.filtered_matches_topic,
                    "keypoints": args.keypoints_topic,
                    "cmd_vel": args.cmd_vel_topic,
                    "tcp_vel": args.tcp_vel_topic,
                    "goal": args.goal_topic,
                    "filter_unc": args.filter_unc_topic,
                    "filter_status": args.filter_status_topic,
                    "filter_update_status": args.filter_update_status_topic,
                    "filter_active_count": args.filter_active_count_topic,
                    "control_wx_debug": args.control_wx_debug_topic,
                },
            }
        )

        self.create_subscription(Matches, args.raw_matches_topic, self.on_raw_matches, qos_profile_sensor_data)
        self.create_subscription(Matches, args.filtered_matches_topic, self.on_filtered_matches, qos_profile_sensor_data)
        self.create_subscription(Keypoints, args.keypoints_topic, self.on_keypoints, qos_profile_sensor_data)
        self.create_subscription(Twist, args.cmd_vel_topic, self.on_cmd_vel, qos_profile_sensor_data)
        self.create_subscription(TwistStamped, args.tcp_vel_topic, self.on_tcp_vel, qos_profile_sensor_data)
        self.create_subscription(Bool, args.goal_topic, self.on_goal, qos_profile_sensor_data)
        self.create_subscription(Float32, args.filter_unc_topic, self.on_filter_unc, qos_profile_sensor_data)
        self.create_subscription(String, args.filter_status_topic, self.on_filter_status, qos_profile_sensor_data)
        self.create_subscription(
            String,
            args.filter_update_status_topic,
            self.on_update_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(UInt32, args.filter_active_count_topic, self.on_active_count, qos_profile_sensor_data)
        self.create_subscription(
            String,
            args.control_wx_debug_topic,
            self.on_control_wx_debug,
            qos_profile_sensor_data,
        )

        period = 1.0 / max(1e-3, float(args.sample_hz))
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(f"Logging compact JSONL to: {self.out_path}")

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def write_event(self, payload: dict):
        self.fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if (self.sample_count % 10) == 0:
            self.fh.flush()

    @staticmethod
    def _safe_age_ms(now_sec: float, recv_sec: float) -> Optional[float]:
        if recv_sec <= 0.0:
            return None
        return max(0.0, 1000.0 * (now_sec - recv_sec))

    @staticmethod
    def _summarize_depth(depth_m: np.ndarray, min_depth_m: float) -> tuple[int, float]:
        if depth_m.size <= 0:
            return 0, math.nan
        valid = np.isfinite(depth_m) & (depth_m > min_depth_m)
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            return 0, math.nan
        med = float(np.median(depth_m[valid]))
        return valid_count, med

    def _to_match_stats(self, msg: Matches) -> MatchStats:
        n = min(len(msg.ref_id), len(msg.xy) // 2)
        stats = MatchStats(n=max(0, n), recv_sec=self.now_sec())
        if n <= 0:
            return stats

        if len(msg.depth_m) >= n:
            depth = np.asarray(msg.depth_m[:n], dtype=np.float32)
        else:
            depth = np.asarray(msg.depth_m, dtype=np.float32)
        valid_count, med = self._summarize_depth(depth, self.args.min_valid_depth_m)
        stats.depth_valid = valid_count
        stats.depth_ratio = (float(valid_count) / float(n)) if n > 0 else 0.0
        stats.depth_median_m = med
        return stats

    def _to_keypoint_stats(self, msg: Keypoints) -> KeypointStats:
        n = len(msg.xy) // 2
        stats = KeypointStats(n=max(0, n), recv_sec=self.now_sec())
        if n <= 0 or len(msg.depth_m) <= 0:
            return stats

        depth = np.asarray(msg.depth_m[:n], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > self.args.min_valid_depth_m)
        valid_count = int(np.count_nonzero(valid))
        stats.depth_valid = valid_count
        stats.depth_ratio = float(valid_count) / float(n)
        return stats

    def on_raw_matches(self, msg: Matches):
        self.raw = self._to_match_stats(msg)

    def on_filtered_matches(self, msg: Matches):
        self.filtered = self._to_match_stats(msg)

    def on_keypoints(self, msg: Keypoints):
        self.kp = self._to_keypoint_stats(msg)

    def on_cmd_vel(self, msg: Twist):
        self.cmd = TwistStats(
            lx=float(msg.linear.x),
            ly=float(msg.linear.y),
            lz=float(msg.linear.z),
            wx=float(msg.angular.x),
            wy=float(msg.angular.y),
            wz=float(msg.angular.z),
            recv_sec=self.now_sec(),
        )

    def on_tcp_vel(self, msg: TwistStamped):
        t = msg.twist
        self.tcp = TwistStats(
            lx=float(t.linear.x),
            ly=float(t.linear.y),
            lz=float(t.linear.z),
            wx=float(t.angular.x),
            wy=float(t.angular.y),
            wz=float(t.angular.z),
            recv_sec=self.now_sec(),
        )

    def on_goal(self, msg: Bool):
        self.goal_reached = bool(msg.data)

    def on_filter_unc(self, msg: Float32):
        self.filter_uncertainty = float(msg.data)

    def on_filter_status(self, msg: String):
        self.filter_status = str(msg.data)

    def on_update_status(self, msg: String):
        self.update_status = str(msg.data)

    def on_active_count(self, msg: UInt32):
        self.active_count = int(msg.data)

    def on_control_wx_debug(self, msg: String):
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                self.ctrl_wx_debug = payload
                self.ctrl_wx_debug_recv_sec = self.now_sec()
        except Exception:
            return

    @staticmethod
    def _twist_dict(now_sec: float, tw: TwistStats) -> dict:
        lin_norm = float(math.sqrt(tw.lx * tw.lx + tw.ly * tw.ly + tw.lz * tw.lz))
        ang_norm = float(math.sqrt(tw.wx * tw.wx + tw.wy * tw.wy + tw.wz * tw.wz))
        return {
            "lx": tw.lx,
            "ly": tw.ly,
            "lz": tw.lz,
            "wx": tw.wx,
            "wy": tw.wy,
            "wz": tw.wz,
            "lin_norm": lin_norm,
            "ang_norm": ang_norm,
            "age_ms": IbvsDebugJsonLogger._safe_age_ms(now_sec, tw.recv_sec),
        }

    @staticmethod
    def _match_dict(now_sec: float, m: MatchStats) -> dict:
        return {
            "n": m.n,
            "depth_valid": m.depth_valid,
            "depth_ratio": m.depth_ratio,
            "depth_med_m": m.depth_median_m if math.isfinite(m.depth_median_m) else None,
            "age_ms": IbvsDebugJsonLogger._safe_age_ms(now_sec, m.recv_sec),
        }

    @staticmethod
    def _kp_dict(now_sec: float, k: KeypointStats) -> dict:
        return {
            "n": k.n,
            "depth_valid": k.depth_valid,
            "depth_ratio": k.depth_ratio,
            "age_ms": IbvsDebugJsonLogger._safe_age_ms(now_sec, k.recv_sec),
        }

    def _update_wx_jitter_counters(self):
        wx = self.cmd.wx
        if abs(wx) <= self.args.wx_near_zero_eps:
            self.wx_near_zero_samples += 1

        if abs(wx) < self.args.wx_sign_eps:
            sign = 0
        else:
            sign = 1 if wx > 0.0 else -1

        if sign != 0 and self.prev_wx_sign != 0 and sign != self.prev_wx_sign:
            self.wx_sign_flips += 1

        if sign != 0:
            self.prev_wx_sign = sign

    def on_timer(self):
        now_sec = self.now_sec()
        rel_ros_sec = now_sec - self.start_ros_sec
        rel_wall_sec = time.monotonic() - self.start_mono_sec
        self._update_wx_jitter_counters()

        sample = {
            "type": "sample",
            "t": round(rel_wall_sec, 3),
            "t_ros": round(rel_ros_sec, 3),
            "cmd": self._twist_dict(now_sec, self.cmd),
            "tcp": self._twist_dict(now_sec, self.tcp),
            "raw": self._match_dict(now_sec, self.raw),
            "filtered": self._match_dict(now_sec, self.filtered),
            "keypoints": self._kp_dict(now_sec, self.kp),
            "goal": self.goal_reached,
            "filter_unc": self.filter_uncertainty if math.isfinite(self.filter_uncertainty) else None,
            "filter_status": self.filter_status or None,
            "update_status": self.update_status or None,
            "active_count": self.active_count,
            "control_wx_debug": self.ctrl_wx_debug,
            "control_wx_debug_age_ms": self._safe_age_ms(now_sec, self.ctrl_wx_debug_recv_sec),
        }

        self.sample_count += 1
        self.write_event(sample)

        if self.args.duration_sec > 0.0 and rel_wall_sec >= self.args.duration_sec:
            self.get_logger().info("Duration reached, stopping logger.")
            rclpy.shutdown()

    def close(self):
        elapsed = time.time() - self.start_wall_sec
        summary = {
            "type": "summary",
            "utc_end": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 3),
            "samples": self.sample_count,
            "sample_hz": self.args.sample_hz,
            "wx_sign_flips": self.wx_sign_flips,
            "wx_near_zero_samples": self.wx_near_zero_samples,
            "wx_near_zero_ratio": (
                float(self.wx_near_zero_samples) / float(self.sample_count)
                if self.sample_count > 0
                else 0.0
            ),
        }
        try:
            self.write_event(summary)
            self.fh.flush()
        finally:
            self.fh.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Compact IBVS JSONL logger for jitter debugging")
    p.add_argument("--out-jsonl", default=f"ros_ws/logs/debug/ibvs_debug_{ts}.jsonl")
    p.add_argument("--sample-hz", type=float, default=5.0)
    p.add_argument("--duration-sec", type=float, default=0.0, help="0 = run until Ctrl-C")
    p.add_argument("--min-valid-depth-m", type=float, default=0.05)
    p.add_argument("--wx-sign-eps", type=float, default=2e-4)
    p.add_argument("--wx-near-zero-eps", type=float, default=5e-4)

    p.add_argument("--raw-matches-topic", default="/ibvs/matches")
    p.add_argument("--filtered-matches-topic", default="/ibvs/filtered_features")
    p.add_argument("--keypoints-topic", default="/ibvs/keypoints")
    p.add_argument("--cmd-vel-topic", default="/cartesian_twist_passthrough_controller/cmd_vel")
    p.add_argument("--tcp-vel-topic", default="/tcp_pose_broadcaster/velocity")
    p.add_argument("--goal-topic", default="/ibvs/control/goal_reached")
    p.add_argument("--filter-unc-topic", default="/ibvs/filter/uncertainty")
    p.add_argument("--filter-status-topic", default="/ibvs/filter/status")
    p.add_argument("--filter-update-status-topic", default="/ibvs/filter/update_status")
    p.add_argument("--filter-active-count-topic", default="/ibvs/filter/active_count")
    p.add_argument("--control-wx-debug-topic", default="/ibvs/control/wx_debug")

    args = p.parse_args(argv)
    if args.sample_hz <= 0.0:
        p.error("--sample-hz must be > 0")
    if args.min_valid_depth_m < 0.0:
        p.error("--min-valid-depth-m must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    rclpy.init(args=None)
    node = IbvsDebugJsonLogger(args)

    def _signal_handler(_sig, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
