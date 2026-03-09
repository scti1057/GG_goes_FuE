#!/usr/bin/env python3
"""
Benchmark RealSense camera throughput.

Supports two data sources:
1) direct: read frames directly from camera via pyrealsense2
2) ros: read ROS image topics from running realsense_driver

Auto mode chooses ros when camera topics are active, otherwise direct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def enum_name(value: Any) -> str:
    text = str(value)
    if "." in text:
        return text.split(".")[-1]
    return text


def args_to_json_dict(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def scenarios_from_args(args: argparse.Namespace) -> List[str]:
    if args.mode == "both":
        if args.both_order == "rgb-rgbd":
            return ["rgb", "rgbd"]
        return ["rgbd", "rgb"]
    return [args.mode]


def safe_device_info(device: Any, camera_info_key: Any) -> Optional[str]:
    try:
        if device.supports(camera_info_key):
            return device.get_info(camera_info_key)
    except Exception:
        return None
    return None


def read_rs_metadata(frame: Any, rs: Any) -> Dict[str, Any]:
    keys = {
        "sensor_timestamp": "sensor_timestamp",
        "frame_timestamp": "frame_timestamp",
        "backend_timestamp": "backend_timestamp",
        "time_of_arrival": "time_of_arrival",
        "frame_counter": "frame_counter",
        "actual_fps": "actual_fps",
    }
    out: Dict[str, Any] = {}
    for key, out_name in keys.items():
        enum_value = getattr(rs.frame_metadata_value, key, None)
        if enum_value is None:
            continue
        try:
            if frame.supports_frame_metadata(enum_value):
                out[out_name] = frame.get_frame_metadata(enum_value)
        except Exception:
            continue
    return out


def list_active_streams(active_profile: Any) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    for stream_profile in active_profile.get_streams():
        entry: Dict[str, Any] = {
            "stream": enum_name(stream_profile.stream_type()),
            "format": enum_name(stream_profile.format()),
            "fps": int(stream_profile.fps()),
        }
        try:
            video = stream_profile.as_video_stream_profile()
            entry["width"] = int(video.width())
            entry["height"] = int(video.height())
        except Exception:
            pass
        streams.append(entry)
    return streams


def write_interval_row(
    interval_writer: csv.DictWriter,
    interval_file,
    row: Dict[str, Any],
) -> None:
    interval_writer.writerow(row)
    interval_file.flush()


def write_frame_row(
    frame_writer: Optional[csv.DictWriter],
    frame_file,
    row: Dict[str, Any],
) -> None:
    if frame_writer is None:
        return
    frame_writer.writerow(row)
    frame_file.flush()


def build_interval_row(
    run_id: str,
    source: str,
    scenario: str,
    elapsed_s: float,
    interval_s: float,
    color_interval_frames: int,
    depth_interval_frames: int,
    total_color_frames: int,
    total_depth_frames: int,
    dropped_color_frames_est: int,
    dropped_depth_frames_est: Optional[int],
    non_monotonic_color_frame_numbers: Optional[int],
    missing_depth_frames: Optional[int],
    capture_timeouts: int,
) -> Dict[str, Any]:
    color_interval_fps = color_interval_frames / max(1e-9, interval_s)
    depth_interval_fps = (
        depth_interval_frames / max(1e-9, interval_s) if depth_interval_frames >= 0 else None
    )
    avg_color_fps = total_color_frames / max(1e-9, elapsed_s)
    avg_depth_fps = (
        total_depth_frames / max(1e-9, elapsed_s) if total_depth_frames >= 0 else None
    )
    return {
        "run_id": run_id,
        "source": source,
        "scenario": scenario,
        "wall_time_utc": utc_now_iso(),
        "elapsed_s": round(elapsed_s, 6),
        "interval_s": round(interval_s, 6),
        "color_interval_frames": color_interval_frames,
        "color_interval_fps": round(color_interval_fps, 6),
        "depth_interval_frames": depth_interval_frames if depth_interval_frames >= 0 else "",
        "depth_interval_fps": round(depth_interval_fps, 6) if depth_interval_fps is not None else "",
        "total_color_frames": total_color_frames,
        "total_depth_frames": total_depth_frames if total_depth_frames >= 0 else "",
        "avg_color_fps": round(avg_color_fps, 6),
        "avg_depth_fps": round(avg_depth_fps, 6) if avg_depth_fps is not None else "",
        "dropped_color_frames_est": dropped_color_frames_est,
        "dropped_depth_frames_est": (
            dropped_depth_frames_est if dropped_depth_frames_est is not None else ""
        ),
        "non_monotonic_color_frame_numbers": (
            non_monotonic_color_frame_numbers
            if non_monotonic_color_frame_numbers is not None
            else ""
        ),
        "missing_depth_frames": missing_depth_frames if missing_depth_frames is not None else "",
        "capture_timeouts": capture_timeouts,
    }


def summarize_color_fps_drift(interval_rows: List[Dict[str, Any]], capture_duration: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    first_cut = capture_duration * 0.2
    last_cut = capture_duration * 0.8
    first_window = [
        float(r["color_interval_fps"])
        for r in interval_rows
        if float(r["elapsed_s"]) <= first_cut
    ]
    last_window = [
        float(r["color_interval_fps"])
        for r in interval_rows
        if float(r["elapsed_s"]) >= last_cut
    ]
    fps_first_avg = avg(first_window)
    fps_last_avg = avg(last_window)
    fps_drift_pct = None
    if fps_first_avg and fps_first_avg > 0 and fps_last_avg is not None:
        fps_drift_pct = 100.0 * (fps_last_avg - fps_first_avg) / fps_first_avg
    return fps_first_avg, fps_last_avg, fps_drift_pct


def collect_scenario_direct(
    rs: Any,
    scenario: str,
    args: argparse.Namespace,
    run_id: str,
    interval_writer: csv.DictWriter,
    interval_file,
    frame_writer: Optional[csv.DictWriter],
    frame_file,
) -> Dict[str, Any]:
    enable_depth = scenario == "rgbd"
    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if enable_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print(
        f"[direct/{scenario}] start: {args.width}x{args.height}@{args.fps} "
        f"(depth={'on' if enable_depth else 'off'})",
        flush=True,
    )

    try:
        active_profile = pipeline.start(config)
    except Exception as exc:
        raise RuntimeError(f"Could not start RealSense stream ({scenario}): {exc}") from exc

    device = active_profile.get_device()
    camera_info = {
        "name": safe_device_info(device, rs.camera_info.name),
        "serial_number": safe_device_info(device, rs.camera_info.serial_number),
        "firmware_version": safe_device_info(device, rs.camera_info.firmware_version),
        "product_line": safe_device_info(device, rs.camera_info.product_line),
    }
    active_streams = list_active_streams(active_profile)

    warmup_frames_color = 0
    warmup_frames_depth = 0
    warmup_timeouts = 0
    timeout_ms = int(max(0.1, args.frame_timeout_sec) * 1000.0)

    if args.warmup_sec > 0.0 and not args.skip_warmup:
        warmup_end = time.monotonic() + args.warmup_sec
        while time.monotonic() < warmup_end:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except RuntimeError:
                warmup_timeouts += 1
                continue
            if frames.get_color_frame():
                warmup_frames_color += 1
            if enable_depth and frames.get_depth_frame():
                warmup_frames_depth += 1

    started_at = utc_now_iso()
    start_mono = time.monotonic()
    start_wall_ns = time.time_ns()
    next_report_mono = start_mono + args.interval_sec
    last_report_mono = start_mono

    total_color_frames = 0
    total_depth_frames = 0 if enable_depth else -1
    interval_color_frames = 0
    interval_depth_frames = 0
    capture_timeouts = 0

    dropped_color_frames_est = 0
    dropped_depth_frames_est = 0 if enable_depth else None
    non_monotonic_color_frame_numbers = 0
    missing_depth_frames = 0 if enable_depth else None

    prev_color_frame_number = None
    prev_depth_frame_number = None
    prev_host_mono = None
    prev_color_ts_ms = None
    prev_depth_ts_ms = None

    host_color_dts: List[float] = []
    cam_color_dts: List[float] = []
    cam_depth_dts: List[float] = []
    interval_rows: List[Dict[str, Any]] = []

    first_color_ts_ms = None
    last_color_ts_ms = None
    first_depth_ts_ms = None
    last_depth_ts_ms = None
    timestamp_domain = None

    try:
        while True:
            now_mono = time.monotonic()
            elapsed = now_mono - start_mono
            if elapsed >= args.duration_sec:
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except RuntimeError:
                capture_timeouts += 1
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            depth_frame = frames.get_depth_frame() if enable_depth else None

            now_mono = time.monotonic()
            elapsed = now_mono - start_mono
            total_color_frames += 1
            interval_color_frames += 1

            color_frame_number = int(color_frame.get_frame_number())
            color_ts_ms = float(color_frame.get_timestamp())
            if timestamp_domain is None:
                timestamp_domain = enum_name(color_frame.get_frame_timestamp_domain())
            if first_color_ts_ms is None:
                first_color_ts_ms = color_ts_ms
            last_color_ts_ms = color_ts_ms

            if prev_color_frame_number is not None:
                gap = color_frame_number - prev_color_frame_number
                if gap > 1:
                    dropped_color_frames_est += gap - 1
                elif gap <= 0:
                    non_monotonic_color_frame_numbers += 1
            prev_color_frame_number = color_frame_number

            if prev_host_mono is not None:
                dt = now_mono - prev_host_mono
                if dt > 0:
                    host_color_dts.append(dt)
            prev_host_mono = now_mono

            if prev_color_ts_ms is not None:
                cam_dt = (color_ts_ms - prev_color_ts_ms) / 1000.0
                if cam_dt > 0:
                    cam_color_dts.append(cam_dt)
            prev_color_ts_ms = color_ts_ms

            if enable_depth:
                if depth_frame:
                    total_depth_frames += 1
                    interval_depth_frames += 1
                    depth_frame_number = int(depth_frame.get_frame_number())
                    depth_ts_ms = float(depth_frame.get_timestamp())
                    if first_depth_ts_ms is None:
                        first_depth_ts_ms = depth_ts_ms
                    last_depth_ts_ms = depth_ts_ms
                    if prev_depth_frame_number is not None:
                        gap_d = depth_frame_number - prev_depth_frame_number
                        if gap_d > 1 and dropped_depth_frames_est is not None:
                            dropped_depth_frames_est += gap_d - 1
                    prev_depth_frame_number = depth_frame_number

                    if prev_depth_ts_ms is not None:
                        cam_d_dt = (depth_ts_ms - prev_depth_ts_ms) / 1000.0
                        if cam_d_dt > 0:
                            cam_depth_dts.append(cam_d_dt)
                    prev_depth_ts_ms = depth_ts_ms
                else:
                    if missing_depth_frames is not None:
                        missing_depth_frames += 1

            metadata = read_rs_metadata(color_frame, rs)
            write_frame_row(
                frame_writer,
                frame_file,
                {
                    "run_id": run_id,
                    "source": "direct",
                    "scenario": scenario,
                    "host_wall_ns": time.time_ns(),
                    "elapsed_s": round(elapsed, 6),
                    "stream": "color",
                    "topic": "",
                    "frame_number": color_frame_number,
                    "msg_stamp_ns": "",
                    "color_timestamp_ms": round(color_ts_ms, 6),
                    "timestamp_domain": timestamp_domain,
                    "width": args.width,
                    "height": args.height,
                    "encoding": "bgr8",
                    "depth_present": int(bool(depth_frame)) if enable_depth else "",
                    "sensor_timestamp_ms": metadata.get("sensor_timestamp", ""),
                    "backend_timestamp_ms": metadata.get("backend_timestamp", ""),
                    "time_of_arrival_ms": metadata.get("time_of_arrival", ""),
                    "actual_fps": metadata.get("actual_fps", ""),
                },
            )

            if now_mono >= next_report_mono:
                report_dt = max(1e-9, now_mono - last_report_mono)
                row = build_interval_row(
                    run_id=run_id,
                    source="direct",
                    scenario=scenario,
                    elapsed_s=elapsed,
                    interval_s=report_dt,
                    color_interval_frames=interval_color_frames,
                    depth_interval_frames=interval_depth_frames if enable_depth else -1,
                    total_color_frames=total_color_frames,
                    total_depth_frames=total_depth_frames,
                    dropped_color_frames_est=dropped_color_frames_est,
                    dropped_depth_frames_est=dropped_depth_frames_est,
                    non_monotonic_color_frame_numbers=non_monotonic_color_frame_numbers,
                    missing_depth_frames=missing_depth_frames,
                    capture_timeouts=capture_timeouts,
                )
                write_interval_row(interval_writer, interval_file, row)
                interval_rows.append(row)
                print(
                    f"[direct/{scenario}] t={elapsed:7.2f}s  "
                    f"color={float(row['color_interval_fps']):6.2f}  "
                    f"avg={float(row['avg_color_fps']):6.2f}  "
                    f"drop~{dropped_color_frames_est}",
                    flush=True,
                )
                interval_color_frames = 0
                interval_depth_frames = 0
                last_report_mono = now_mono
                while next_report_mono <= now_mono:
                    next_report_mono += args.interval_sec

        end_mono = time.monotonic()
        capture_duration = end_mono - start_mono

        if interval_color_frames > 0 or (enable_depth and interval_depth_frames > 0):
            report_dt = max(1e-9, end_mono - last_report_mono)
            elapsed = capture_duration
            row = build_interval_row(
                run_id=run_id,
                source="direct",
                scenario=scenario,
                elapsed_s=elapsed,
                interval_s=report_dt,
                color_interval_frames=interval_color_frames,
                depth_interval_frames=interval_depth_frames if enable_depth else -1,
                total_color_frames=total_color_frames,
                total_depth_frames=total_depth_frames,
                dropped_color_frames_est=dropped_color_frames_est,
                dropped_depth_frames_est=dropped_depth_frames_est,
                non_monotonic_color_frame_numbers=non_monotonic_color_frame_numbers,
                missing_depth_frames=missing_depth_frames,
                capture_timeouts=capture_timeouts,
            )
            write_interval_row(interval_writer, interval_file, row)
            interval_rows.append(row)

        sorted_host_color_dts = sorted(host_color_dts)
        sorted_cam_color_dts = sorted(cam_color_dts)
        sorted_cam_depth_dts = sorted(cam_depth_dts)
        fps_first_avg, fps_last_avg, fps_drift_pct = summarize_color_fps_drift(
            interval_rows, capture_duration
        )

        summary = {
            "source": "direct",
            "scenario": scenario,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now_iso(),
            "requested_config": {
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "enable_depth": enable_depth,
                "duration_sec": args.duration_sec,
                "warmup_sec": 0.0 if args.skip_warmup else args.warmup_sec,
            },
            "camera_info": camera_info,
            "active_streams": active_streams,
            "stats": {
                "capture_duration_sec": capture_duration,
                "total_color_frames": total_color_frames,
                "total_depth_frames": total_depth_frames if enable_depth else None,
                "avg_color_fps": total_color_frames / max(1e-9, capture_duration),
                "avg_depth_fps": (
                    total_depth_frames / max(1e-9, capture_duration) if enable_depth else None
                ),
                "avg_fps_primary": total_color_frames / max(1e-9, capture_duration),
                "dropped_color_frames_est": dropped_color_frames_est,
                "dropped_depth_frames_est": dropped_depth_frames_est if enable_depth else None,
                "non_monotonic_color_frame_numbers": non_monotonic_color_frame_numbers,
                "missing_depth_frames": missing_depth_frames if enable_depth else None,
                "capture_timeouts": capture_timeouts,
                "warmup_frames_color": warmup_frames_color,
                "warmup_frames_depth": warmup_frames_depth if enable_depth else None,
                "warmup_timeouts": warmup_timeouts,
                "host_color_dt_sec_p50": percentile(sorted_host_color_dts, 50.0),
                "host_color_dt_sec_p90": percentile(sorted_host_color_dts, 90.0),
                "host_color_dt_sec_p99": percentile(sorted_host_color_dts, 99.0),
                "cam_color_dt_sec_p50": percentile(sorted_cam_color_dts, 50.0),
                "cam_color_dt_sec_p90": percentile(sorted_cam_color_dts, 90.0),
                "cam_color_dt_sec_p99": percentile(sorted_cam_color_dts, 99.0),
                "cam_depth_dt_sec_p50": percentile(sorted_cam_depth_dts, 50.0)
                if enable_depth
                else None,
                "cam_depth_dt_sec_p90": percentile(sorted_cam_depth_dts, 90.0)
                if enable_depth
                else None,
                "cam_depth_dt_sec_p99": percentile(sorted_cam_depth_dts, 99.0)
                if enable_depth
                else None,
                "fps_first20_avg": fps_first_avg,
                "fps_last20_avg": fps_last_avg,
                "fps_drift_pct_first20_to_last20": fps_drift_pct,
            },
            "timestamps": {
                "timestamp_domain": timestamp_domain,
                "first_color_timestamp_ms": first_color_ts_ms,
                "last_color_timestamp_ms": last_color_ts_ms,
                "color_timestamp_span_sec": (
                    (last_color_ts_ms - first_color_ts_ms) / 1000.0
                    if (first_color_ts_ms is not None and last_color_ts_ms is not None)
                    else None
                ),
                "first_depth_timestamp_ms": first_depth_ts_ms if enable_depth else None,
                "last_depth_timestamp_ms": last_depth_ts_ms if enable_depth else None,
                "depth_timestamp_span_sec": (
                    (last_depth_ts_ms - first_depth_ts_ms) / 1000.0
                    if (
                        enable_depth
                        and first_depth_ts_ms is not None
                        and last_depth_ts_ms is not None
                    )
                    else None
                ),
                "host_start_wall_ns": start_wall_ns,
                "host_end_wall_ns": time.time_ns(),
            },
        }
        print(
            f"[direct/{scenario}] done: color_avg={summary['stats']['avg_color_fps']:.2f} fps",
            flush=True,
        )
        return summary
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def probe_ros_color_topic(args: argparse.Namespace) -> bool:
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ImportError:
        return False

    rclpy.init(args=None)
    got_msg = {"value": False}
    node_name = f"rs_bench_probe_{int(time.time() * 1000) % 1000000}"
    node = rclpy.create_node(node_name)

    def on_color(_msg: Any) -> None:
        got_msg["value"] = True

    sub = node.create_subscription(
        Image, args.ros_color_topic, on_color, qos_profile_sensor_data
    )

    deadline = time.monotonic() + args.ros_probe_timeout_sec
    while time.monotonic() < deadline and not got_msg["value"]:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    return bool(got_msg["value"])


def collect_scenario_ros(
    scenario: str,
    args: argparse.Namespace,
    run_id: str,
    interval_writer: csv.DictWriter,
    interval_file,
    frame_writer: Optional[csv.DictWriter],
    frame_file,
) -> Dict[str, Any]:
    try:
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise RuntimeError(
            "ROS mode requested but rclpy/sensor_msgs are unavailable in this environment."
        ) from exc

    enable_depth = scenario == "rgbd"
    expected_dt = 1.0 / args.fps if args.fps > 0 else None

    if not rclpy.ok():
        rclpy.init(args=None)
    node_name = f"rs_bench_ros_{scenario}_{int(time.time() * 1000) % 1000000}"
    node = rclpy.create_node(node_name)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    state_lock = threading.Lock()

    started_at = utc_now_iso()
    start_mono = None
    start_wall_ns = None

    total_color_frames = 0
    total_depth_frames = 0 if enable_depth else -1
    interval_color_frames = 0
    interval_depth_frames = 0
    capture_timeouts = 0

    dropped_color_frames_est = 0
    dropped_depth_frames_est = 0 if enable_depth else None

    prev_host_color_mono = None
    prev_color_stamp_sec = None
    prev_depth_stamp_sec = None
    first_color_stamp_sec = None
    last_color_stamp_sec = None
    first_depth_stamp_sec = None
    last_depth_stamp_sec = None

    host_color_dts: List[float] = []
    ros_color_stamp_dts: List[float] = []
    ros_depth_stamp_dts: List[float] = []
    interval_rows: List[Dict[str, Any]] = []

    latest_color_mono = None
    latest_depth_mono = None

    def to_stamp_sec(msg: Any) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def on_color(msg: Any) -> None:
        nonlocal total_color_frames
        nonlocal interval_color_frames
        nonlocal prev_host_color_mono
        nonlocal prev_color_stamp_sec
        nonlocal first_color_stamp_sec
        nonlocal last_color_stamp_sec
        nonlocal dropped_color_frames_est
        nonlocal latest_color_mono
        nonlocal start_mono
        nonlocal start_wall_ns

        now_mono = time.monotonic()
        stamp_sec = to_stamp_sec(msg)
        with state_lock:
            if start_mono is None:
                start_mono = now_mono
                start_wall_ns = time.time_ns()
            latest_color_mono = now_mono

            total_color_frames += 1
            interval_color_frames += 1
            if first_color_stamp_sec is None:
                first_color_stamp_sec = stamp_sec
            last_color_stamp_sec = stamp_sec

            if prev_host_color_mono is not None:
                host_dt = now_mono - prev_host_color_mono
                if host_dt > 0:
                    host_color_dts.append(host_dt)
            prev_host_color_mono = now_mono

            if prev_color_stamp_sec is not None:
                ros_dt = stamp_sec - prev_color_stamp_sec
                if ros_dt > 0:
                    ros_color_stamp_dts.append(ros_dt)
                    if expected_dt is not None and ros_dt > expected_dt * 1.5:
                        missed = max(0, int(round(ros_dt / expected_dt)) - 1)
                        dropped_color_frames_est += missed
            prev_color_stamp_sec = stamp_sec

        write_frame_row(
            frame_writer,
            frame_file,
            {
                "run_id": run_id,
                "source": "ros",
                "scenario": scenario,
                "host_wall_ns": time.time_ns(),
                "elapsed_s": round(now_mono - start_mono, 6) if start_mono else 0.0,
                "stream": "color",
                "topic": args.ros_color_topic,
                "frame_number": "",
                "msg_stamp_ns": int(stamp_sec * 1e9),
                "color_timestamp_ms": "",
                "timestamp_domain": "ROS_TIME",
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": msg.encoding,
                "depth_present": "",
                "sensor_timestamp_ms": "",
                "backend_timestamp_ms": "",
                "time_of_arrival_ms": "",
                "actual_fps": "",
            },
        )

    def on_depth(msg: Any) -> None:
        nonlocal total_depth_frames
        nonlocal interval_depth_frames
        nonlocal prev_depth_stamp_sec
        nonlocal first_depth_stamp_sec
        nonlocal last_depth_stamp_sec
        nonlocal dropped_depth_frames_est
        nonlocal latest_depth_mono

        now_mono = time.monotonic()
        stamp_sec = to_stamp_sec(msg)
        with state_lock:
            latest_depth_mono = now_mono
            total_depth_frames += 1
            interval_depth_frames += 1
            if first_depth_stamp_sec is None:
                first_depth_stamp_sec = stamp_sec
            last_depth_stamp_sec = stamp_sec

            if prev_depth_stamp_sec is not None:
                ros_dt = stamp_sec - prev_depth_stamp_sec
                if ros_dt > 0:
                    ros_depth_stamp_dts.append(ros_dt)
                    if expected_dt is not None and ros_dt > expected_dt * 1.5:
                        missed = max(0, int(round(ros_dt / expected_dt)) - 1)
                        if dropped_depth_frames_est is not None:
                            dropped_depth_frames_est += missed
            prev_depth_stamp_sec = stamp_sec

        write_frame_row(
            frame_writer,
            frame_file,
            {
                "run_id": run_id,
                "source": "ros",
                "scenario": scenario,
                "host_wall_ns": time.time_ns(),
                "elapsed_s": round(now_mono - start_mono, 6) if start_mono else 0.0,
                "stream": "depth",
                "topic": args.ros_depth_topic,
                "frame_number": "",
                "msg_stamp_ns": int(stamp_sec * 1e9),
                "color_timestamp_ms": "",
                "timestamp_domain": "ROS_TIME",
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": msg.encoding,
                "depth_present": 1,
                "sensor_timestamp_ms": "",
                "backend_timestamp_ms": "",
                "time_of_arrival_ms": "",
                "actual_fps": "",
            },
        )

    sub_color = node.create_subscription(
        Image,
        args.ros_color_topic,
        on_color,
        qos_profile_sensor_data,
        callback_group=ReentrantCallbackGroup(),
    )
    sub_depth = None
    if enable_depth:
        sub_depth = node.create_subscription(
            Image,
            args.ros_depth_topic,
            on_depth,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

    try:
        print(
            f"[ros/{scenario}] start: color={args.ros_color_topic} "
            + (f"depth={args.ros_depth_topic}" if enable_depth else "depth=off"),
            flush=True,
        )

        if args.warmup_sec > 0.0 and not args.skip_warmup:
            warmup_end = time.monotonic() + args.warmup_sec
            while time.monotonic() < warmup_end:
                executor.spin_once(timeout_sec=0.05)

            with state_lock:
                total_color_frames = 0
                interval_color_frames = 0
                dropped_color_frames_est = 0
                prev_host_color_mono = None
                prev_color_stamp_sec = None
                first_color_stamp_sec = None
                last_color_stamp_sec = None
                host_color_dts.clear()
                ros_color_stamp_dts.clear()

                if enable_depth:
                    total_depth_frames = 0
                    interval_depth_frames = 0
                    dropped_depth_frames_est = 0
                    prev_depth_stamp_sec = None
                    first_depth_stamp_sec = None
                    last_depth_stamp_sec = None
                    ros_depth_stamp_dts.clear()

                start_mono = None
                start_wall_ns = None

        wait_start = time.monotonic()
        while True:
            with state_lock:
                has_start = start_mono is not None
            if has_start:
                break
            if (time.monotonic() - wait_start) >= args.ros_start_timeout_sec:
                break
            executor.spin_once(timeout_sec=0.05)
        with state_lock:
            has_start = start_mono is not None
        if not has_start:
            raise RuntimeError(
                f"No messages on color topic '{args.ros_color_topic}' within {args.ros_start_timeout_sec}s."
            )

        with state_lock:
            start_mono_local = start_mono
        next_report_mono = start_mono_local + args.interval_sec
        last_report_mono = start_mono_local

        while True:
            now_mono = time.monotonic()
            elapsed = now_mono - start_mono_local
            if elapsed >= args.duration_sec:
                break

            executor.spin_once(timeout_sec=0.05)

            now_mono = time.monotonic()
            elapsed = now_mono - start_mono_local
            with state_lock:
                if (
                    latest_color_mono is not None
                    and (now_mono - latest_color_mono) > args.frame_timeout_sec
                ):
                    capture_timeouts += 1
                    latest_color_mono = now_mono
                if (
                    enable_depth
                    and latest_depth_mono is not None
                    and (now_mono - latest_depth_mono) > args.frame_timeout_sec
                ):
                    capture_timeouts += 1
                    latest_depth_mono = now_mono

            if now_mono >= next_report_mono:
                report_dt = max(1e-9, now_mono - last_report_mono)
                with state_lock:
                    total_color_frames_local = total_color_frames
                    total_depth_frames_local = total_depth_frames
                    interval_color_frames_local = interval_color_frames
                    interval_depth_frames_local = interval_depth_frames
                    dropped_color_frames_est_local = dropped_color_frames_est
                    dropped_depth_frames_est_local = dropped_depth_frames_est

                row = build_interval_row(
                    run_id=run_id,
                    source="ros",
                    scenario=scenario,
                    elapsed_s=elapsed,
                    interval_s=report_dt,
                    color_interval_frames=interval_color_frames_local,
                    depth_interval_frames=interval_depth_frames_local if enable_depth else -1,
                    total_color_frames=total_color_frames_local,
                    total_depth_frames=total_depth_frames_local,
                    dropped_color_frames_est=dropped_color_frames_est_local,
                    dropped_depth_frames_est=dropped_depth_frames_est_local,
                    non_monotonic_color_frame_numbers=None,
                    missing_depth_frames=(
                        max(0, total_color_frames_local - total_depth_frames_local)
                        if enable_depth
                        else None
                    ),
                    capture_timeouts=capture_timeouts,
                )
                write_interval_row(interval_writer, interval_file, row)
                interval_rows.append(row)
                print(
                    f"[ros/{scenario}] t={elapsed:7.2f}s  "
                    f"color={float(row['color_interval_fps']):6.2f}  "
                    f"avg={float(row['avg_color_fps']):6.2f}",
                    flush=True,
                )
                with state_lock:
                    interval_color_frames = 0
                    interval_depth_frames = 0
                last_report_mono = now_mono
                while next_report_mono <= now_mono:
                    next_report_mono += args.interval_sec

        end_mono = time.monotonic()
        capture_duration = end_mono - start_mono_local
        with state_lock:
            interval_color_frames_local = interval_color_frames
            interval_depth_frames_local = interval_depth_frames
            total_color_frames_local = total_color_frames
            total_depth_frames_local = total_depth_frames
            dropped_color_frames_est_local = dropped_color_frames_est
            dropped_depth_frames_est_local = dropped_depth_frames_est
            first_color_stamp_sec_local = first_color_stamp_sec
            last_color_stamp_sec_local = last_color_stamp_sec
            first_depth_stamp_sec_local = first_depth_stamp_sec
            last_depth_stamp_sec_local = last_depth_stamp_sec
            host_color_dts_local = list(host_color_dts)
            ros_color_stamp_dts_local = list(ros_color_stamp_dts)
            ros_depth_stamp_dts_local = list(ros_depth_stamp_dts)
            start_wall_ns_local = start_wall_ns
        if interval_color_frames_local > 0 or (enable_depth and interval_depth_frames_local > 0):
            row = build_interval_row(
                run_id=run_id,
                source="ros",
                scenario=scenario,
                elapsed_s=capture_duration,
                interval_s=max(1e-9, end_mono - last_report_mono),
                color_interval_frames=interval_color_frames_local,
                depth_interval_frames=interval_depth_frames_local if enable_depth else -1,
                total_color_frames=total_color_frames_local,
                total_depth_frames=total_depth_frames_local,
                dropped_color_frames_est=dropped_color_frames_est_local,
                dropped_depth_frames_est=dropped_depth_frames_est_local,
                non_monotonic_color_frame_numbers=None,
                missing_depth_frames=(
                    max(0, total_color_frames_local - total_depth_frames_local)
                    if enable_depth
                    else None
                ),
                capture_timeouts=capture_timeouts,
            )
            write_interval_row(interval_writer, interval_file, row)
            interval_rows.append(row)

        if enable_depth and args.ros_require_depth and total_depth_frames_local == 0:
            raise RuntimeError(
                f"RGBD requested, but no messages received on depth topic '{args.ros_depth_topic}'."
            )

        sorted_host_color_dts = sorted(host_color_dts_local)
        sorted_ros_color_dts = sorted(ros_color_stamp_dts_local)
        sorted_ros_depth_dts = sorted(ros_depth_stamp_dts_local)
        fps_first_avg, fps_last_avg, fps_drift_pct = summarize_color_fps_drift(
            interval_rows, capture_duration
        )

        summary = {
            "source": "ros",
            "scenario": scenario,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now_iso(),
            "requested_config": {
                "mode": "topic",
                "duration_sec": args.duration_sec,
                "warmup_sec": 0.0 if args.skip_warmup else args.warmup_sec,
                "color_topic": args.ros_color_topic,
                "depth_topic": args.ros_depth_topic if enable_depth else None,
            },
            "camera_info": None,
            "active_streams": None,
            "stats": {
                "capture_duration_sec": capture_duration,
                "total_color_frames": total_color_frames_local,
                "total_depth_frames": total_depth_frames_local if enable_depth else None,
                "avg_color_fps": total_color_frames_local / max(1e-9, capture_duration),
                "avg_depth_fps": (
                    total_depth_frames_local / max(1e-9, capture_duration)
                    if enable_depth
                    else None
                ),
                "avg_fps_primary": total_color_frames_local / max(1e-9, capture_duration),
                "dropped_color_frames_est": dropped_color_frames_est_local,
                "dropped_depth_frames_est": dropped_depth_frames_est_local
                if enable_depth
                else None,
                "non_monotonic_color_frame_numbers": None,
                "missing_depth_frames": (
                    max(0, total_color_frames_local - total_depth_frames_local)
                    if enable_depth
                    else None
                ),
                "capture_timeouts": capture_timeouts,
                "host_color_dt_sec_p50": percentile(sorted_host_color_dts, 50.0),
                "host_color_dt_sec_p90": percentile(sorted_host_color_dts, 90.0),
                "host_color_dt_sec_p99": percentile(sorted_host_color_dts, 99.0),
                "ros_color_stamp_dt_sec_p50": percentile(sorted_ros_color_dts, 50.0),
                "ros_color_stamp_dt_sec_p90": percentile(sorted_ros_color_dts, 90.0),
                "ros_color_stamp_dt_sec_p99": percentile(sorted_ros_color_dts, 99.0),
                "ros_depth_stamp_dt_sec_p50": percentile(sorted_ros_depth_dts, 50.0)
                if enable_depth
                else None,
                "ros_depth_stamp_dt_sec_p90": percentile(sorted_ros_depth_dts, 90.0)
                if enable_depth
                else None,
                "ros_depth_stamp_dt_sec_p99": percentile(sorted_ros_depth_dts, 99.0)
                if enable_depth
                else None,
                "fps_first20_avg": fps_first_avg,
                "fps_last20_avg": fps_last_avg,
                "fps_drift_pct_first20_to_last20": fps_drift_pct,
            },
            "timestamps": {
                "timestamp_domain": "ROS_TIME",
                "first_color_timestamp_sec": first_color_stamp_sec_local,
                "last_color_timestamp_sec": last_color_stamp_sec_local,
                "color_timestamp_span_sec": (
                    (last_color_stamp_sec_local - first_color_stamp_sec_local)
                    if (
                        first_color_stamp_sec_local is not None
                        and last_color_stamp_sec_local is not None
                    )
                    else None
                ),
                "first_depth_timestamp_sec": first_depth_stamp_sec_local if enable_depth else None,
                "last_depth_timestamp_sec": last_depth_stamp_sec_local if enable_depth else None,
                "depth_timestamp_span_sec": (
                    (last_depth_stamp_sec_local - first_depth_stamp_sec_local)
                    if (
                        enable_depth
                        and first_depth_stamp_sec_local is not None
                        and last_depth_stamp_sec_local is not None
                    )
                    else None
                ),
                "host_start_wall_ns": start_wall_ns_local,
                "host_end_wall_ns": time.time_ns(),
            },
        }
        print(
            f"[ros/{scenario}] done: color_avg={summary['stats']['avg_color_fps']:.2f} fps",
            flush=True,
        )
        return summary
    finally:
        executor.shutdown(timeout_sec=1.0)
        executor.remove_node(node)
        node.destroy_subscription(sub_color)
        if sub_depth is not None:
            node.destroy_subscription(sub_depth)
        node.destroy_node()


def detect_source(args: argparse.Namespace) -> Tuple[str, str]:
    if args.source in ("direct", "ros"):
        return args.source, "forced by --source"

    ros_active = probe_ros_color_topic(args)
    if ros_active:
        return "ros", f"received messages on {args.ros_color_topic}"
    return "direct", f"no messages on {args.ros_color_topic}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Intel RealSense FPS via ROS topics or direct camera access. "
            "Auto mode uses ROS topics when driver is running, otherwise direct camera access."
        )
    )
    parser.add_argument("--source", choices=["auto", "direct", "ros"], default="auto")
    parser.add_argument("--mode", choices=["rgb", "rgbd", "both"], default="both")
    parser.add_argument(
        "--both-order",
        choices=["rgb-rgbd", "rgbd-rgb"],
        default="rgb-rgbd",
        help="Order used when --mode both.",
    )
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--frame-timeout-sec", type=float, default=2.0)
    parser.add_argument("--serial", default="")
    parser.add_argument("--ros-color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--ros-depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument(
        "--ros-probe-timeout-sec",
        type=float,
        default=2.5,
        help="How long auto mode waits for a color topic message.",
    )
    parser.add_argument(
        "--ros-start-timeout-sec",
        type=float,
        default=10.0,
        help="How long ROS mode waits for the first color frame of a scenario.",
    )
    parser.add_argument(
        "--ros-require-depth",
        action="store_true",
        help="Fail rgbd ROS scenario if depth topic delivers no frames.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/camera_benchmark"),
        help="Output directory for CSV/JSON logs.",
    )
    parser.add_argument("--log-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.duration_sec <= 0.0:
        print("ERROR: --duration-sec must be > 0", file=sys.stderr)
        return 2
    if args.interval_sec <= 0.0:
        print("ERROR: --interval-sec must be > 0", file=sys.stderr)
        return 2
    if args.ros_probe_timeout_sec <= 0.0:
        print("ERROR: --ros-probe-timeout-sec must be > 0", file=sys.stderr)
        return 2
    if args.ros_start_timeout_sec <= 0.0:
        print("ERROR: --ros-start-timeout-sec must be > 0", file=sys.stderr)
        return 2

    selected_source, source_reason = detect_source(args)
    print(f"Selected source: {selected_source} ({source_reason})", flush=True)

    rs = None
    if selected_source == "direct":
        try:
            import pyrealsense2 as rs_import
        except ImportError:
            print(
                "ERROR: Direct mode selected, but pyrealsense2 is not installed.",
                file=sys.stderr,
            )
            return 3
        rs = rs_import

        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
        except Exception as exc:
            print(f"ERROR: Could not query RealSense devices: {exc}", file=sys.stderr)
            return 4
        if not devices:
            print("ERROR: No RealSense device detected for direct mode.", file=sys.stderr)
            return 4

    run_id = datetime.now().strftime("realsense_bench_%Y%m%d_%H%M%S")
    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    interval_path = log_dir / f"{run_id}_intervals.csv"
    summary_path = log_dir / f"{run_id}_summary.json"
    frame_path = log_dir / f"{run_id}_frames.csv"

    print(f"Run ID: {run_id}", flush=True)
    print(f"Logs dir: {log_dir.resolve()}", flush=True)

    scenarios = scenarios_from_args(args)
    summaries: List[Dict[str, Any]] = []

    with interval_path.open("w", newline="", encoding="utf-8") as interval_file:
        interval_writer = csv.DictWriter(
            interval_file,
            fieldnames=[
                "run_id",
                "source",
                "scenario",
                "wall_time_utc",
                "elapsed_s",
                "interval_s",
                "color_interval_frames",
                "color_interval_fps",
                "depth_interval_frames",
                "depth_interval_fps",
                "total_color_frames",
                "total_depth_frames",
                "avg_color_fps",
                "avg_depth_fps",
                "dropped_color_frames_est",
                "dropped_depth_frames_est",
                "non_monotonic_color_frame_numbers",
                "missing_depth_frames",
                "capture_timeouts",
            ],
        )
        interval_writer.writeheader()
        interval_file.flush()

        frame_file = None
        frame_writer = None
        if args.log_frames:
            frame_file = frame_path.open("w", newline="", encoding="utf-8")
            frame_writer = csv.DictWriter(
                frame_file,
                fieldnames=[
                    "run_id",
                    "source",
                    "scenario",
                    "host_wall_ns",
                    "elapsed_s",
                    "stream",
                    "topic",
                    "frame_number",
                    "msg_stamp_ns",
                    "color_timestamp_ms",
                    "timestamp_domain",
                    "width",
                    "height",
                    "encoding",
                    "depth_present",
                    "sensor_timestamp_ms",
                    "backend_timestamp_ms",
                    "time_of_arrival_ms",
                    "actual_fps",
                ],
            )
            frame_writer.writeheader()
            frame_file.flush()

        try:
            for scenario in scenarios:
                if selected_source == "direct":
                    summaries.append(
                        collect_scenario_direct(
                            rs=rs,
                            scenario=scenario,
                            args=args,
                            run_id=run_id,
                            interval_writer=interval_writer,
                            interval_file=interval_file,
                            frame_writer=frame_writer,
                            frame_file=frame_file,
                        )
                    )
                else:
                    summaries.append(
                        collect_scenario_ros(
                            scenario=scenario,
                            args=args,
                            run_id=run_id,
                            interval_writer=interval_writer,
                            interval_file=interval_file,
                            frame_writer=frame_writer,
                            frame_file=frame_file,
                        )
                    )
        finally:
            if frame_file is not None:
                frame_file.close()

    if selected_source == "ros":
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    by_scenario = {s["scenario"]: s for s in summaries}
    comparison = None
    if "rgb" in by_scenario and "rgbd" in by_scenario:
        rgb_fps = by_scenario["rgb"]["stats"]["avg_fps_primary"]
        rgbd_fps = by_scenario["rgbd"]["stats"]["avg_fps_primary"]
        comparison = {
            "metric": "avg_fps_primary",
            "avg_fps_primary_rgb": rgb_fps,
            "avg_fps_primary_rgbd": rgbd_fps,
            "delta_rgbd_minus_rgb": rgbd_fps - rgb_fps,
            "delta_percent_vs_rgb": ((rgbd_fps - rgb_fps) / rgb_fps * 100.0)
            if rgb_fps > 0
            else None,
            "dropped_color_frames_est_rgb": by_scenario["rgb"]["stats"][
                "dropped_color_frames_est"
            ],
            "dropped_color_frames_est_rgbd": by_scenario["rgbd"]["stats"][
                "dropped_color_frames_est"
            ],
        }

    report = {
        "run_id": run_id,
        "started_at_utc": summaries[0]["started_at_utc"] if summaries else utc_now_iso(),
        "ended_at_utc": utc_now_iso(),
        "script": "realsense_camera_benchmark.py",
        "selected_source": selected_source,
        "source_reason": source_reason,
        "args": args_to_json_dict(args),
        "interval_csv": str(interval_path.resolve()),
        "frame_csv": str(frame_path.resolve()) if args.log_frames else None,
        "scenarios": summaries,
        "comparison": comparison,
        "notes": [
            "auto source selection uses ROS topic activity on ros_color_topic.",
            "direct mode can fail if another process already owns the camera device.",
            "ROS mode measures topic throughput, not direct USB camera throughput.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Summary JSON: {summary_path.resolve()}", flush=True)
    print(f"Intervals CSV: {interval_path.resolve()}", flush=True)
    if args.log_frames:
        print(f"Frames CSV:   {frame_path.resolve()}", flush=True)
    if comparison is not None:
        print(
            f"Compare RGB vs RGBD ({comparison['metric']}): "
            f"{comparison['avg_fps_primary_rgb']:.2f} -> {comparison['avg_fps_primary_rgbd']:.2f} "
            f"(delta {comparison['delta_rgbd_minus_rgb']:.2f}, "
            f"{comparison['delta_percent_vs_rgb']:.2f}%)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
