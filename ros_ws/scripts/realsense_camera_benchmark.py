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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def fps_to_rgb(fps: float, target_fps: float) -> Tuple[int, int, int]:
    if target_fps <= 0.0:
        target_fps = 60.0
    ratio = clamp(fps / target_fps, 0.0, 1.0)
    # 0.0 -> red, 0.5 -> yellow, 1.0 -> green
    if ratio <= 0.5:
        t = ratio / 0.5
        r = 255
        g = int(255 * t)
        b = 0
    else:
        t = (ratio - 0.5) / 0.5
        r = int(255 * (1.0 - t))
        g = 255
        b = 0
    return r, g, b


def colorize_fps(
    fps_value: Optional[float],
    target_fps: float,
    use_color: bool,
    width: int = 7,
) -> str:
    if fps_value is None:
        return "-".rjust(width)
    text = f"{fps_value:6.2f}"
    if not use_color:
        return text.rjust(width)
    r, g, b = fps_to_rgb(fps_value, target_fps)
    # Use dark text on bright background for readability.
    return f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;20;20;20m{text}\x1b[0m"


def _short_case_name(summary: Dict[str, Any]) -> str:
    if "sweep_case" in summary:
        return str(summary["sweep_case"])
    scenario = str(summary.get("scenario", ""))
    if "__" in scenario:
        return scenario.split("__", 1)[1]
    return scenario


def print_performance_matrix(
    summaries: List[Dict[str, Any]],
    target_fps: float,
    use_color: bool,
) -> None:
    if not summaries:
        return

    headers = [
        "Case",
        "Source",
        "AvgColor",
        "AvgDepth",
        "DropColor",
        "DropDepth",
        "Timeouts",
    ]
    rows: List[List[str]] = []
    for s in summaries:
        stats = s.get("stats", {})
        avg_color = stats.get("avg_color_fps")
        avg_depth = stats.get("avg_depth_fps")
        drop_color = stats.get("dropped_color_frames_est")
        drop_depth = stats.get("dropped_depth_frames_est")
        timeouts = stats.get("capture_timeouts")
        rows.append(
            [
                _short_case_name(s),
                str(s.get("source", "")),
                colorize_fps(avg_color, target_fps, use_color),
                colorize_fps(avg_depth, target_fps, use_color),
                str(drop_color if drop_color is not None else "-"),
                str(drop_depth if drop_depth is not None else "-"),
                str(timeouts if timeouts is not None else "-"),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            # ANSI sequences do not add visual width; keep width based on plain fallback lengths.
            plain = cell
            if "\x1b[" in cell:
                plain = " 00.00 "
            col_widths[i] = max(col_widths[i], len(plain))

    def fmt_cell(i: int, value: str) -> str:
        if i in (2, 3):  # colored fps columns
            return value.rjust(col_widths[i])
        return value.ljust(col_widths[i])

    sep = " | "
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    divider = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\nPerformance Matrix (target FPS: {:.1f})".format(target_fps), flush=True)
    print(header_line, flush=True)
    print(divider, flush=True)
    for row in rows:
        print(sep.join(fmt_cell(i, c) for i, c in enumerate(row)), flush=True)


def write_performance_matrix_csv(
    summaries: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "scenario",
                "source",
                "sweep_case",
                "depth_topic_used",
                "avg_color_fps",
                "avg_depth_fps",
                "dropped_color_frames_est",
                "dropped_depth_frames_est",
                "capture_timeouts",
                "fps_first20_avg",
                "fps_last20_avg",
                "fps_drift_pct_first20_to_last20",
            ],
        )
        writer.writeheader()
        for s in summaries:
            stats = s.get("stats", {})
            writer.writerow(
                {
                    "case": _short_case_name(s),
                    "scenario": s.get("scenario", ""),
                    "source": s.get("source", ""),
                    "sweep_case": s.get("sweep_case", ""),
                    "depth_topic_used": s.get("depth_topic_used", ""),
                    "avg_color_fps": stats.get("avg_color_fps", ""),
                    "avg_depth_fps": stats.get("avg_depth_fps", ""),
                    "dropped_color_frames_est": stats.get("dropped_color_frames_est", ""),
                    "dropped_depth_frames_est": stats.get("dropped_depth_frames_est", ""),
                    "capture_timeouts": stats.get("capture_timeouts", ""),
                    "fps_first20_avg": stats.get("fps_first20_avg", ""),
                    "fps_last20_avg": stats.get("fps_last20_avg", ""),
                    "fps_drift_pct_first20_to_last20": stats.get(
                        "fps_drift_pct_first20_to_last20", ""
                    ),
                }
            )


def scenarios_from_args(args: argparse.Namespace) -> List[str]:
    if args.mode == "both":
        if args.both_order == "rgb-rgbd":
            return ["rgb", "rgbd"]
        return ["rgbd", "rgb"]
    return [args.mode]


def ros_image_topic_transport(topic: str) -> str:
    if topic.endswith("/compressedDepth"):
        return "compressedDepth"
    if topic.endswith("/compressed"):
        return "compressed"
    return "raw"


def ros_topic_uses_compressed_msg(topic: str) -> bool:
    return ros_image_topic_transport(topic) in ("compressed", "compressedDepth")


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
    enable_depth = scenario.startswith("rgbd")
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
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError:
        return False

    rclpy.init(args=None)
    got_msg = {"value": False}
    node_name = f"rs_bench_probe_{int(time.time() * 1000) % 1000000}"
    node = rclpy.create_node(node_name)

    def on_color(_msg: Any) -> None:
        got_msg["value"] = True

    color_msg_type = (
        CompressedImage
        if ros_topic_uses_compressed_msg(args.ros_color_topic)
        else Image
    )
    sub = node.create_subscription(
        color_msg_type, args.ros_color_topic, on_color, qos_profile_sensor_data
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
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError as exc:
        raise RuntimeError(
            "ROS mode requested but rclpy/sensor_msgs are unavailable in this environment."
        ) from exc

    enable_depth = scenario.startswith("rgbd")
    expected_dt = 1.0 / args.fps if args.fps > 0 else None
    color_transport = ros_image_topic_transport(args.ros_color_topic)
    depth_transport = ros_image_topic_transport(args.ros_depth_topic) if enable_depth else None
    color_msg_is_compressed = ros_topic_uses_compressed_msg(args.ros_color_topic)
    depth_msg_is_compressed = (
        ros_topic_uses_compressed_msg(args.ros_depth_topic) if enable_depth else False
    )
    color_msg_type = CompressedImage if color_msg_is_compressed else Image
    depth_msg_type = CompressedImage if depth_msg_is_compressed else Image

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
                "width": "" if color_msg_is_compressed else int(msg.width),
                "height": "" if color_msg_is_compressed else int(msg.height),
                "encoding": (
                    str(getattr(msg, "format", "compressed"))
                    if color_msg_is_compressed
                    else msg.encoding
                ),
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
                "width": "" if depth_msg_is_compressed else int(msg.width),
                "height": "" if depth_msg_is_compressed else int(msg.height),
                "encoding": (
                    str(getattr(msg, "format", "compressed"))
                    if depth_msg_is_compressed
                    else msg.encoding
                ),
                "depth_present": 1,
                "sensor_timestamp_ms": "",
                "backend_timestamp_ms": "",
                "time_of_arrival_ms": "",
                "actual_fps": "",
            },
        )

    sub_color = node.create_subscription(
        color_msg_type,
        args.ros_color_topic,
        on_color,
        qos_profile_sensor_data,
        callback_group=ReentrantCallbackGroup(),
    )
    sub_depth = None
    if enable_depth:
        sub_depth = node.create_subscription(
            depth_msg_type,
            args.ros_depth_topic,
            on_depth,
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

    try:
        print(
            f"[ros/{scenario}] start: color={args.ros_color_topic} ({color_transport}) "
            + (
                f"depth={args.ros_depth_topic} ({depth_transport})"
                if enable_depth
                else "depth=off"
            ),
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
                "color_transport": color_transport,
                "depth_topic": args.ros_depth_topic if enable_depth else None,
                "depth_transport": depth_transport if enable_depth else None,
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


def _normalize_node_name(name: str) -> str:
    n = name.strip()
    if not n:
        return n
    if not n.startswith("/"):
        n = "/" + n
    return n


def _parameter_value_to_python(value_msg: Any, parameter_type: Any) -> Any:
    if value_msg.type == parameter_type.PARAMETER_BOOL:
        return bool(value_msg.bool_value)
    if value_msg.type == parameter_type.PARAMETER_INTEGER:
        return int(value_msg.integer_value)
    if value_msg.type == parameter_type.PARAMETER_DOUBLE:
        return float(value_msg.double_value)
    if value_msg.type == parameter_type.PARAMETER_STRING:
        return str(value_msg.string_value)
    if value_msg.type == parameter_type.PARAMETER_BYTE_ARRAY:
        return list(value_msg.byte_array_value)
    if value_msg.type == parameter_type.PARAMETER_BOOL_ARRAY:
        return list(value_msg.bool_array_value)
    if value_msg.type == parameter_type.PARAMETER_INTEGER_ARRAY:
        return list(value_msg.integer_array_value)
    if value_msg.type == parameter_type.PARAMETER_DOUBLE_ARRAY:
        return list(value_msg.double_array_value)
    if value_msg.type == parameter_type.PARAMETER_STRING_ARRAY:
        return list(value_msg.string_array_value)
    return None


def resolve_ros_camera_node(requested_node: str, timeout_sec: float) -> str:
    if requested_node.strip():
        return _normalize_node_name(requested_node)

    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError("rclpy is required to auto-discover camera node.") from exc

    started_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        started_here = True

    node_name = f"rs_bench_node_discovery_{int(time.time() * 1000) % 1000000}"
    node = rclpy.create_node(node_name)
    try:
        deadline = time.monotonic() + max(0.2, timeout_sec)
        candidates: List[str] = []
        while time.monotonic() < deadline:
            names = node.get_node_names_and_namespaces()
            candidates = []
            for name, ns in names:
                if ns == "/":
                    full = f"/{name}"
                else:
                    full = f"{ns}/{name}"
                full = full.replace("//", "/")
                candidates.append(full)
            if candidates:
                break
            rclpy.spin_once(node, timeout_sec=0.05)

        preferred = "/camera/camera"
        if preferred in candidates:
            return preferred

        for full in candidates:
            if full.endswith("/camera"):
                return full

        if candidates:
            raise RuntimeError(
                "Could not auto-detect RealSense camera node. "
                f"Discovered nodes include: {', '.join(sorted(candidates)[:10])}. "
                "Pass --ros-camera-node explicitly."
            )
        raise RuntimeError("No ROS nodes discovered while trying to auto-detect camera node.")
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()


def ros_get_parameters(
    target_node: str,
    param_names: List[str],
    timeout_sec: float,
) -> Dict[str, Any]:
    try:
        import rclpy
        from rcl_interfaces.srv import GetParameters
        from rcl_interfaces.msg import ParameterType
    except ImportError as exc:
        raise RuntimeError("rclpy/rcl_interfaces are required for ROS parameter access.") from exc

    started_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        started_here = True

    node = rclpy.create_node(f"rs_bench_get_params_{int(time.time() * 1000) % 1000000}")
    try:
        service_name = f"{target_node}/get_parameters"
        client = node.create_client(GetParameters, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"Parameter service '{service_name}' is not available."
            )

        request = GetParameters.Request()
        request.names = list(param_names)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError(
                f"Timeout waiting for response from '{service_name}'."
            )
        if future.exception() is not None:
            raise RuntimeError(
                f"Service call '{service_name}' failed: {future.exception()}"
            )
        result = future.result()
        if result is None or len(result.values) != len(param_names):
            raise RuntimeError(f"Could not read parameters from node '{target_node}'.")

        out: Dict[str, Any] = {}
        missing: List[str] = []
        for name, value_msg in zip(param_names, result.values):
            if value_msg.type == ParameterType.PARAMETER_NOT_SET:
                missing.append(name)
                continue
            out[name] = _parameter_value_to_python(value_msg, ParameterType)

        if missing:
            raise RuntimeError(
                f"Node '{target_node}' is missing parameters: {', '.join(missing)}"
            )
        return out
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()


def _python_to_parameter_message(name: str, value: Any, parameter_type: Any, parameter_cls: Any, parameter_value_cls: Any) -> Any:
    p = parameter_cls()
    p.name = name
    pv = parameter_value_cls()
    if isinstance(value, bool):
        pv.type = parameter_type.PARAMETER_BOOL
        pv.bool_value = bool(value)
    elif isinstance(value, int):
        pv.type = parameter_type.PARAMETER_INTEGER
        pv.integer_value = int(value)
    elif isinstance(value, float):
        pv.type = parameter_type.PARAMETER_DOUBLE
        pv.double_value = float(value)
    elif isinstance(value, str):
        pv.type = parameter_type.PARAMETER_STRING
        pv.string_value = str(value)
    else:
        raise RuntimeError(
            f"Unsupported parameter type for '{name}': {type(value).__name__}. "
            "Supported: bool, int, float, str."
        )
    p.value = pv
    return p


def ros_set_parameters(
    target_node: str,
    values: Dict[str, Any],
    timeout_sec: float,
) -> List[Dict[str, Any]]:
    try:
        import rclpy
        from rcl_interfaces.msg import Parameter as RosParameter
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
    except ImportError as exc:
        raise RuntimeError("rclpy is required for ROS parameter updates.") from exc

    started_here = False
    if not rclpy.ok():
        rclpy.init(args=None)
        started_here = True

    node = rclpy.create_node(f"rs_bench_set_params_{int(time.time() * 1000) % 1000000}")
    try:
        service_name = f"{target_node}/set_parameters"
        client = node.create_client(SetParameters, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"Parameter service '{service_name}' is not available."
            )

        request = SetParameters.Request()
        request.parameters = [
            _python_to_parameter_message(
                name=name,
                value=value,
                parameter_type=ParameterType,
                parameter_cls=RosParameter,
                parameter_value_cls=ParameterValue,
            )
            for name, value in values.items()
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError(
                f"Timeout waiting for response from '{service_name}'."
            )
        if future.exception() is not None:
            raise RuntimeError(
                f"Service call '{service_name}' failed: {future.exception()}"
            )
        result = future.result()
        if result is None:
            raise RuntimeError(f"Could not set parameters on node '{target_node}'.")

        status: List[Dict[str, Any]] = []
        if len(result.results) != len(values):
            raise RuntimeError(
                f"Unexpected SetParameters response length from node '{target_node}'."
            )
        for name, r in zip(values.keys(), result.results):
            status.append(
                {
                    "name": name,
                    "successful": bool(r.successful),
                    "reason": str(r.reason),
                }
            )
        return status
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()


def ros_set_parameters_with_retry(
    target_node: str,
    values: Dict[str, Any],
    timeout_sec: float,
    retries: int,
    retry_wait_sec: float,
) -> List[Dict[str, Any]]:
    attempts = max(1, int(retries))
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return ros_set_parameters(
                target_node=target_node,
                values=values,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                print(
                    f"[sweep] set_parameters attempt {i+1}/{attempts} failed: {exc}. "
                    f"retrying in {retry_wait_sec:.2f}s ...",
                    flush=True,
                )
                if retry_wait_sec > 0.0:
                    time.sleep(retry_wait_sec)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected retry state while setting ROS parameters.")


def build_filter_sweep_cases(include_all_off: bool) -> List[Tuple[str, Dict[str, bool]]]:
    align = "align_depth.enable"
    spatial = "spatial_filter.enable"
    temporal = "temporal_filter.enable"
    hole = "hole_filling_filter.enable"

    cases: List[Tuple[str, Dict[str, bool]]] = [
        ("baseline", {}),
        ("align_off", {align: False}),
        ("spatial_off", {spatial: False}),
        ("temporal_off", {temporal: False}),
        ("hole_filling_off", {hole: False}),
    ]
    if include_all_off:
        cases.append(
            (
                "align_and_all_depth_filters_off",
                {
                    align: False,
                    spatial: False,
                    temporal: False,
                    hole: False,
                },
            )
        )
    return cases


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
    parser.add_argument(
        "--target-fps",
        type=float,
        default=60.0,
        help="Target FPS used for red->green color scale in performance matrix.",
    )
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--frame-timeout-sec", type=float, default=2.0)
    parser.add_argument("--serial", default="")
    parser.add_argument(
        "--ros-color-topic",
        default="/camera/camera/color/image_raw",
        help=(
            "ROS color topic. Supports raw '/.../image_raw' and compressed "
            "'/.../image_raw/compressed'."
        ),
    )
    parser.add_argument(
        "--ros-depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
        help=(
            "ROS depth topic. Supports raw '/.../image_raw' and compressed depth "
            "'/.../image_raw/compressedDepth'."
        ),
    )
    parser.add_argument(
        "--ros-depth-topic-unaligned",
        default="/camera/camera/depth/image_rect_raw",
        help=(
            "Fallback depth topic used during filter sweep when align_depth is disabled "
            "and the aligned depth topic stops publishing. For compressed depth, pass "
            "the matching '/compressedDepth' topic explicitly."
        ),
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
        "--ros-camera-node",
        default="/camera/camera",
        help=(
            "Target camera node for runtime parameter control "
            "(e.g. /camera/camera). Use empty string for auto-discovery."
        ),
    )
    parser.add_argument(
        "--ros-param-timeout-sec",
        type=float,
        default=5.0,
        help="Timeout for ROS parameter get/set service calls.",
    )
    parser.add_argument(
        "--ros-param-retries",
        type=int,
        default=3,
        help="Retry attempts for ROS set_parameters in filter sweep.",
    )
    parser.add_argument(
        "--ros-param-retry-wait-sec",
        type=float,
        default=0.75,
        help="Delay between set_parameters retry attempts.",
    )
    parser.add_argument(
        "--ros-param-settle-sec",
        type=float,
        default=1.0,
        help="Wait time after applying camera parameter changes.",
    )
    parser.add_argument(
        "--ros-filter-sweep",
        action="store_true",
        help=(
            "Run ROS filter impact sweep (baseline + each depth processing toggle). "
            "Forces rgbd scenario with per-case duration from --sweep-duration-sec."
        ),
    )
    parser.add_argument(
        "--sweep-duration-sec",
        type=float,
        default=15.0,
        help="Measurement duration per sweep case (seconds).",
    )
    parser.add_argument(
        "--sweep-include-all-off",
        action="store_true",
        help="Also benchmark a case with align + all depth filters disabled.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/camera_benchmark"),
        help="Output directory for CSV/JSON logs.",
    )
    parser.add_argument("--log-frames", action="store_true")
    parser.add_argument(
        "--color-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ANSI color output for matrix values.",
    )
    return parser.parse_args()


def run_ros_filter_sweep(
    args: argparse.Namespace,
    run_id: str,
    interval_writer: csv.DictWriter,
    interval_file,
    frame_writer: Optional[csv.DictWriter],
    frame_file,
) -> List[Dict[str, Any]]:
    camera_node = resolve_ros_camera_node(args.ros_camera_node, args.ros_param_timeout_sec)
    param_names = [
        "align_depth.enable",
        "spatial_filter.enable",
        "temporal_filter.enable",
        "hole_filling_filter.enable",
    ]
    baseline_values = ros_get_parameters(
        target_node=camera_node,
        param_names=param_names,
        timeout_sec=args.ros_param_timeout_sec,
    )
    print(f"[sweep] camera node: {camera_node}", flush=True)
    print(f"[sweep] baseline params: {baseline_values}", flush=True)

    cases = build_filter_sweep_cases(include_all_off=args.sweep_include_all_off)
    summaries: List[Dict[str, Any]] = []

    sweep_args = argparse.Namespace(**vars(args))
    sweep_args.mode = "rgbd"
    sweep_args.duration_sec = float(args.sweep_duration_sec)
    if sweep_args.duration_sec <= 0.0:
        raise RuntimeError("--sweep-duration-sec must be > 0.")

    try:
        for case_name, overrides in cases:
            target_values = dict(baseline_values)
            target_values.update(overrides)
            try:
                set_status = ros_set_parameters_with_retry(
                    target_node=camera_node,
                    values=target_values,
                    timeout_sec=args.ros_param_timeout_sec,
                    retries=args.ros_param_retries,
                    retry_wait_sec=args.ros_param_retry_wait_sec,
                )
                failed = [s for s in set_status if not s["successful"]]
                if failed:
                    raise RuntimeError(
                        f"Failed to apply camera params for case '{case_name}': {failed}"
                    )
            except Exception as exc:
                print(
                    f"[sweep] WARNING: skipping case '{case_name}' due to parameter update failure: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                summaries.append(
                    {
                        "source": "ros",
                        "scenario": f"rgbd__{case_name}",
                        "sweep_case": case_name,
                        "camera_node": camera_node,
                        "camera_params_before_case": baseline_values,
                        "camera_params_applied": target_values,
                        "depth_topic_used": "",
                        "error": str(exc),
                        "stats": {},
                    }
                )
                continue

            if args.ros_param_settle_sec > 0.0:
                time.sleep(args.ros_param_settle_sec)

            case_args = argparse.Namespace(**vars(sweep_args))
            if (
                overrides.get("align_depth.enable") is False
                and "aligned_depth_to_color" in str(args.ros_depth_topic)
            ):
                case_args.ros_depth_topic = args.ros_depth_topic_unaligned

            scenario_name = f"rgbd__{case_name}"
            print(
                f"[sweep] case={case_name} overrides={overrides} "
                f"depth_topic={case_args.ros_depth_topic} "
                f"duration={sweep_args.duration_sec:.1f}s",
                flush=True,
            )
            summary = collect_scenario_ros(
                scenario=scenario_name,
                args=case_args,
                run_id=run_id,
                interval_writer=interval_writer,
                interval_file=interval_file,
                frame_writer=frame_writer,
                frame_file=frame_file,
            )
            summary["sweep_case"] = case_name
            summary["camera_node"] = camera_node
            summary["camera_params_before_case"] = baseline_values
            summary["camera_params_applied"] = target_values
            summary["depth_topic_used"] = case_args.ros_depth_topic
            summaries.append(summary)
    finally:
        try:
            restore_status = ros_set_parameters_with_retry(
                target_node=camera_node,
                values=baseline_values,
                timeout_sec=args.ros_param_timeout_sec,
                retries=args.ros_param_retries,
                retry_wait_sec=args.ros_param_retry_wait_sec,
            )
            failed_restore = [s for s in restore_status if not s["successful"]]
            if failed_restore:
                print(
                    f"[sweep] WARNING: failed to restore baseline camera params: {failed_restore}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print("[sweep] camera parameters restored to baseline.", flush=True)
        except Exception as exc:
            print(
                f"[sweep] WARNING: failed to restore baseline camera params: {exc}",
                file=sys.stderr,
                flush=True,
            )

    return summaries


def main() -> int:
    args = parse_args()

    if args.duration_sec <= 0.0:
        print("ERROR: --duration-sec must be > 0", file=sys.stderr)
        return 2
    if args.interval_sec <= 0.0:
        print("ERROR: --interval-sec must be > 0", file=sys.stderr)
        return 2
    if args.target_fps <= 0.0:
        print("ERROR: --target-fps must be > 0", file=sys.stderr)
        return 2
    if args.ros_probe_timeout_sec <= 0.0:
        print("ERROR: --ros-probe-timeout-sec must be > 0", file=sys.stderr)
        return 2
    if args.ros_start_timeout_sec <= 0.0:
        print("ERROR: --ros-start-timeout-sec must be > 0", file=sys.stderr)
        return 2
    if args.ros_param_timeout_sec <= 0.0:
        print("ERROR: --ros-param-timeout-sec must be > 0", file=sys.stderr)
        return 2
    if args.ros_param_retries <= 0:
        print("ERROR: --ros-param-retries must be > 0", file=sys.stderr)
        return 2
    if args.ros_param_retry_wait_sec < 0.0:
        print("ERROR: --ros-param-retry-wait-sec must be >= 0", file=sys.stderr)
        return 2
    if args.ros_param_settle_sec < 0.0:
        print("ERROR: --ros-param-settle-sec must be >= 0", file=sys.stderr)
        return 2
    if args.sweep_duration_sec <= 0.0:
        print("ERROR: --sweep-duration-sec must be > 0", file=sys.stderr)
        return 2

    selected_source, source_reason = detect_source(args)
    print(f"Selected source: {selected_source} ({source_reason})", flush=True)
    if args.ros_filter_sweep and selected_source != "ros":
        print(
            "ERROR: --ros-filter-sweep requires source 'ros' (driver running and ROS topics available).",
            file=sys.stderr,
        )
        return 2

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
    matrix_path = log_dir / f"{run_id}_matrix.csv"

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
            if args.ros_filter_sweep:
                summaries.extend(
                    run_ros_filter_sweep(
                        args=args,
                        run_id=run_id,
                        interval_writer=interval_writer,
                        interval_file=interval_file,
                        frame_writer=frame_writer,
                        frame_file=frame_file,
                    )
                )
            else:
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
    if "rgb" in by_scenario and "rgbd" in by_scenario and not args.ros_filter_sweep:
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
        "matrix_csv": str(matrix_path.resolve()),
        "frame_csv": str(frame_path.resolve()) if args.log_frames else None,
        "scenarios": summaries,
        "comparison": comparison,
        "notes": [
            "auto source selection uses ROS topic activity on ros_color_topic.",
            "direct mode can fail if another process already owns the camera device.",
            "ROS mode measures topic throughput, not direct USB camera throughput.",
            "ROS topic mode supports both sensor_msgs/Image and sensor_msgs/CompressedImage topics.",
            "ros-filter-sweep applies runtime camera parameters on realsense node and restores baseline after run.",
        ],
    }

    write_performance_matrix_csv(summaries=summaries, output_path=matrix_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_performance_matrix(
        summaries=summaries,
        target_fps=float(args.target_fps),
        use_color=bool(args.color_output),
    )

    print(f"Summary JSON: {summary_path.resolve()}", flush=True)
    print(f"Intervals CSV: {interval_path.resolve()}", flush=True)
    print(f"Matrix CSV:    {matrix_path.resolve()}", flush=True)
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
