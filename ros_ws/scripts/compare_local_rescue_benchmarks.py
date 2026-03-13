#!/usr/bin/env python3
"""Compare OFF/SHADOW/ACTIVE local-rescue benchmark runs in one table.

Examples:
  python3 ros_ws/scripts/compare_local_rescue_benchmarks.py \
    --off ros_ws/logs/local_rescue_benchmark/bench_off \
    --shadow ros_ws/logs/local_rescue_benchmark/bench_shadow \
    --active ros_ws/logs/local_rescue_benchmark/bench_active
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

from evaluate_local_rescue_benchmark import ModeSummary, summarize_mode


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None:
        return None
    if den <= 1e-9:
        return None
    return float(num) / float(den)


def fnum(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def collect(mode: str, run_dir: str) -> ModeSummary:
    s = summarize_mode(run_dir, mode)
    if s is None:
        raise RuntimeError(
            f"Could not summarize mode '{mode}' from run_dir '{run_dir}'. "
            f"Expected subfolder '{mode}' with *.db3."
        )
    return s


def metric_row(name: str, off_v, sh_v, ac_v, better: str) -> Dict[str, str]:
    return {
        "metric": name,
        "off": str(off_v),
        "shadow": str(sh_v),
        "active": str(ac_v),
        "better": better,
    }


def build_rows(off: ModeSummary, sh: ModeSummary, ac: ModeSummary) -> List[Dict[str, str]]:
    def g(summary: ModeSummary, key: str):
        return summary.counters_delta.get(key)

    def gl(summary: ModeSummary, key: str):
        return summary.counters_last.get(key)

    def gm(summary: ModeSummary, key: str):
        return summary.counters_mean.get(key)

    def fl(summary: ModeSummary, key: str):
        return summary.floats_last.get(key)

    def fm(summary: ModeSummary, key: str):
        return summary.floats_mean.get(key)

    off_attempts = g(off, "attempts")
    sh_attempts = g(sh, "attempts")
    ac_attempts = g(ac, "attempts")

    off_success = g(off, "success")
    sh_success = g(sh, "success")
    ac_success = g(ac, "success")

    off_reject = g(off, "reject")
    sh_reject = g(sh, "reject")
    ac_reject = g(ac, "reject")

    off_success_rate = safe_div(off_success, off_attempts)
    sh_success_rate = safe_div(sh_success, sh_attempts)
    ac_success_rate = safe_div(ac_success, ac_attempts)

    off_success_ps = safe_div(off_success, off.duration_sec)
    sh_success_ps = safe_div(sh_success, sh.duration_sec)
    ac_success_ps = safe_div(ac_success, ac.duration_sec)

    off_update_ps = safe_div(g(off, "update_success_count"), off.duration_sec)
    sh_update_ps = safe_div(g(sh, "update_success_count"), sh.duration_sec)
    ac_update_ps = safe_div(g(ac, "update_success_count"), ac.duration_sec)

    rows = [
        metric_row("duration_s", fnum(off.duration_sec, 2), fnum(sh.duration_sec, 2), fnum(ac.duration_sec, 2), "match similar durations"),
        metric_row("attempts_delta", off_attempts, sh_attempts, ac_attempts, "context"),
        metric_row("success_delta", off_success, sh_success, ac_success, "higher"),
        metric_row("reject_delta", off_reject, sh_reject, ac_reject, "lower"),
        metric_row("success_rate_delta", fnum(off_success_rate), fnum(sh_success_rate), fnum(ac_success_rate), "higher"),
        metric_row("success_per_sec", fnum(off_success_ps), fnum(sh_success_ps), fnum(ac_success_ps), "higher"),
        metric_row("update_success_count_delta", g(off, "update_success_count"), g(sh, "update_success_count"), g(ac, "update_success_count"), "higher"),
        metric_row("update_success_per_sec", fnum(off_update_ps), fnum(sh_update_ps), fnum(ac_update_ps), "higher"),
        metric_row("active_count_mean", fnum(gm(off, "active_count")), fnum(gm(sh, "active_count")), fnum(gm(ac, "active_count")), "higher (until saturation)"),
        metric_row("active_count_last", gl(off, "active_count"), gl(sh, "active_count"), gl(ac, "active_count"), "context"),
        metric_row("filter_uncertainty_mean", fnum(fm(off, "uncertainty")), fnum(fm(sh, "uncertainty")), fnum(fm(ac, "uncertainty")), "lower"),
        metric_row("filter_uncertainty_last", fnum(fl(off, "uncertainty")), fnum(fl(sh, "uncertainty")), fnum(fl(ac, "uncertainty")), "lower"),
        metric_row("matches_rate_hz", fnum(off.rates.get("matches_rate_hz")), fnum(sh.rates.get("matches_rate_hz")), fnum(ac.rates.get("matches_rate_hz")), "context"),
        metric_row("filtered_rate_hz", fnum(off.rates.get("filtered_rate_hz")), fnum(sh.rates.get("filtered_rate_hz")), fnum(ac.rates.get("filtered_rate_hz")), "context"),
    ]
    return rows


def rows_to_markdown(rows: List[Dict[str, str]]) -> str:
    headers = ["metric", "off", "shadow", "active", "better"]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append(
            "| "
            + " | ".join(str(r[h]).replace("|", "\\|") for h in headers)
            + " |"
        )
    return "\n".join(out)


def write_csv(rows: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "off", "shadow", "active", "better"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Compare OFF/SHADOW/ACTIVE benchmark runs.")
    parser.add_argument("--off", required=True, help="run dir for OFF benchmark")
    parser.add_argument("--shadow", required=True, help="run dir for SHADOW benchmark")
    parser.add_argument("--active", required=True, help="run dir for ACTIVE benchmark")
    parser.add_argument(
        "--out-md",
        default="",
        help="optional output markdown file path",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="optional output csv file path",
    )
    args = parser.parse_args()

    off = collect("off", os.path.abspath(args.off))
    shadow = collect("shadow", os.path.abspath(args.shadow))
    active = collect("active", os.path.abspath(args.active))

    rows = build_rows(off, shadow, active)
    md = rows_to_markdown(rows)

    print("\nBenchmark comparison")
    print(md)

    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("Benchmark comparison\n\n")
            f.write(md)
            f.write("\n")
        print(f"\nWrote markdown: {os.path.abspath(args.out_md)}")

    if args.out_csv:
        write_csv(rows, os.path.abspath(args.out_csv))
        print(f"Wrote csv: {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
