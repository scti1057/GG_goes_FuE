#!/usr/bin/env python3
"""Summarize local-rescue benchmark runs from rosbag2 sqlite db3 files.

Reads one benchmark run directory, e.g.
  ros_ws/logs/local_rescue_benchmark/bench_off
and prints per-mode summary for off/shadow/active.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from typing import Dict, Optional


COUNTER_TOPICS = {
    "attempts": "/ibvs/matching/local_rescue_attempts",
    "success": "/ibvs/matching/local_rescue_success",
    "reject": "/ibvs/matching/local_rescue_reject",
    "active_count": "/ibvs/filter/active_count",
    "update_success_count": "/ibvs/filter/update_success_count",
}
CUMULATIVE_COUNTER_KEYS = {"attempts", "success", "reject", "update_success_count"}

FLOAT_TOPICS = {
    "uncertainty": "/ibvs/filter/uncertainty",
}

RATE_TOPICS = {
    "matches_rate_hz": "/ibvs/matches",
    "filtered_rate_hz": "/ibvs/filtered_features",
}

PARAM_ORDER = [
    "sim_floor",
    "use_adaptive_gates",
    "adaptive_radius_min_px",
    "adaptive_radius_max_px",
    "adaptive_sim_threshold_min",
    "adaptive_sim_threshold_max",
    "kp_sigma_low_px",
    "kp_sigma_high_px",
    "local_ambiguity_min_score_gap",
    "local_ambiguity_min_score_ratio",
]


@dataclass
class ModeSummary:
    mode: str
    duration_sec: float
    message_count: int
    counters_last: Dict[str, Optional[int]]
    counters_delta: Dict[str, Optional[int]]
    counters_mean: Dict[str, Optional[float]]
    floats_last: Dict[str, Optional[float]]
    floats_mean: Dict[str, Optional[float]]
    rates: Dict[str, float]
    params: Dict[str, object]


def decode_uint32(blob: bytes) -> int:
    return int(struct.unpack_from("<I", blob, 4)[0])


def decode_float32(blob: bytes) -> float:
    return float(struct.unpack_from("<f", blob, 4)[0])


def parse_params_file(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        return {}

    out: Dict[str, object] = {}
    value_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("mode="):
                out["mode"] = line.split("=", 1)[1]
                continue
            value_lines.append(line)

    def parse_value(line: str):
        m = re.search(r":\s*(.+)$", line)
        if not m:
            return line
        val = m.group(1).strip()
        low = line.lower()
        if "boolean" in low:
            return val.lower() == "true"
        try:
            if "." in val or "e" in val.lower():
                return float(val)
            return int(val)
        except Exception:
            return val

    for key, line in zip(PARAM_ORDER, value_lines):
        out[key] = parse_value(line)

    return out


def summarize_mode(run_dir: str, mode: str) -> Optional[ModeSummary]:
    mode_dir = os.path.join(run_dir, mode)
    if not os.path.isdir(mode_dir):
        return None

    db_files = sorted(glob.glob(os.path.join(mode_dir, "*.db3")))
    if not db_files:
        return None
    db_path = db_files[0]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    topics = cur.fetchall()
    name_to_id = {name: tid for tid, name, _ in topics}

    cur.execute("SELECT COUNT(*) FROM messages")
    message_count = int(cur.fetchone()[0])

    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
    min_ts, max_ts = cur.fetchone()
    duration_sec = 0.0
    if min_ts is not None and max_ts is not None and max_ts >= min_ts:
        duration_sec = (max_ts - min_ts) * 1e-9

    cur.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id")
    counts = {int(topic_id): int(cnt) for topic_id, cnt in cur.fetchall()}

    counters_last: Dict[str, Optional[int]] = {}
    counters_delta: Dict[str, Optional[int]] = {}
    counters_mean: Dict[str, Optional[float]] = {}
    for key, topic in COUNTER_TOPICS.items():
        tid = name_to_id.get(topic)
        if tid is None:
            counters_last[key] = None
            counters_delta[key] = None
            counters_mean[key] = None
            continue
        cur.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
            (tid,),
        )
        first_row = cur.fetchone()
        cur.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1",
            (tid,),
        )
        last_row = cur.fetchone()
        if not last_row:
            counters_last[key] = None
            counters_delta[key] = None
            counters_mean[key] = None
            continue

        v_last = decode_uint32(last_row[0])
        counters_last[key] = v_last

        if first_row and key in CUMULATIVE_COUNTER_KEYS:
            v_first = decode_uint32(first_row[0])
            counters_delta[key] = int(v_last - v_first)
        else:
            counters_delta[key] = None

        if key == "active_count":
            cur.execute("SELECT data FROM messages WHERE topic_id=?", (tid,))
            vals = [decode_uint32(row[0]) for row in cur.fetchall()]
            counters_mean[key] = (sum(vals) / len(vals)) if vals else None
        else:
            counters_mean[key] = None

    floats_last: Dict[str, Optional[float]] = {}
    floats_mean: Dict[str, Optional[float]] = {}
    for key, topic in FLOAT_TOPICS.items():
        tid = name_to_id.get(topic)
        if tid is None:
            floats_last[key] = None
            floats_mean[key] = None
            continue
        cur.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1",
            (tid,),
        )
        row = cur.fetchone()
        floats_last[key] = decode_float32(row[0]) if row else None
        cur.execute("SELECT data FROM messages WHERE topic_id=?", (tid,))
        vals = [decode_float32(r[0]) for r in cur.fetchall()]
        floats_mean[key] = (sum(vals) / len(vals)) if vals else None

    rates: Dict[str, float] = {}
    for key, topic in RATE_TOPICS.items():
        tid = name_to_id.get(topic)
        topic_count = counts.get(tid, 0) if tid is not None else 0
        rates[key] = (topic_count / duration_sec) if duration_sec > 1e-6 else 0.0

    conn.close()

    params_path = os.path.join(run_dir, f"{mode}_params.txt")
    params = parse_params_file(params_path)

    return ModeSummary(
        mode=mode,
        duration_sec=duration_sec,
        message_count=message_count,
        counters_last=counters_last,
        counters_delta=counters_delta,
        counters_mean=counters_mean,
        floats_last=floats_last,
        floats_mean=floats_mean,
        rates=rates,
        params=params,
    )


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(summary: ModeSummary):
    attempts_last = summary.counters_last.get("attempts")
    success_last = summary.counters_last.get("success")
    reject_last = summary.counters_last.get("reject")
    attempts_delta = summary.counters_delta.get("attempts")
    success_delta = summary.counters_delta.get("success")
    reject_delta = summary.counters_delta.get("reject")

    success_rate = None
    if attempts_delta and attempts_delta > 0 and success_delta is not None:
        success_rate = float(success_delta) / float(attempts_delta)

    print(f"\n=== {summary.mode.upper()} ===")
    print(f"duration_s: {summary.duration_sec:.2f}")
    print(f"message_count: {summary.message_count}")
    print(f"attempts_last: {fmt(attempts_last)}")
    print(f"success_last: {fmt(success_last)}")
    print(f"reject_last: {fmt(reject_last)}")
    print(f"attempts_delta: {fmt(attempts_delta)}")
    print(f"success_delta: {fmt(success_delta)}")
    print(f"reject_delta: {fmt(reject_delta)}")
    print(f"success_rate_delta: {fmt(success_rate)}")
    print(f"active_count_last: {fmt(summary.counters_last.get('active_count'))}")
    print(f"active_count_mean: {fmt(summary.counters_mean.get('active_count'))}")
    print(
        "update_success_count_delta: "
        f"{fmt(summary.counters_delta.get('update_success_count'))}"
    )
    print(f"filter_uncertainty_last: {fmt(summary.floats_last.get('uncertainty'))}")
    print(f"filter_uncertainty_mean: {fmt(summary.floats_mean.get('uncertainty'))}")
    print(f"matches_rate_hz: {fmt(summary.rates.get('matches_rate_hz'))}")
    print(f"filtered_rate_hz: {fmt(summary.rates.get('filtered_rate_hz'))}")

    if summary.params:
        print("params:")
        for key in PARAM_ORDER:
            if key in summary.params:
                print(f"  {key}: {summary.params[key]}")


def main():
    parser = argparse.ArgumentParser(description="Summarize local rescue benchmark run.")
    parser.add_argument(
        "run_dir",
        help=(
            "Path to one run directory, e.g. "
            "ros_ws/logs/local_rescue_benchmark/bench_off"
        ),
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir not found: {run_dir}")

    print(f"Run dir: {run_dir}")

    found = False
    for mode in ("off", "shadow", "active"):
        summary = summarize_mode(run_dir, mode)
        if summary is None:
            continue
        found = True
        print_summary(summary)

    if not found:
        raise SystemExit("No mode folders with db3 found (expected off/shadow/active).")


if __name__ == "__main__":
    main()
