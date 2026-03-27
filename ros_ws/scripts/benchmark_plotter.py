#!/usr/bin/env python3
"""Create analysis and comparison plots from benchmark CSV runs.

Usage examples:
  python3 /home/ros_ws/scripts/benchmark_plotter.py
  python3 /home/ros_ws/scripts/benchmark_plotter.py --bench-dir /home/ros_ws/logs/benchmark_csv/bench_20260327_152320
  python3 /home/ros_ws/scripts/benchmark_plotter.py --bench-dir ... --out-dir /home/ros_ws/logs/benchmark_csv/bench_.../analysis
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None


STAGE_DIR_TO_KEY = {
    "stage_1_keypoint_only": "stage1",
    "stage_2_filter": "stage2",
    "stage_3_filter_local_rescue": "stage3",
}

DEFAULT_METRICS: list[tuple[str, str]] = [
    ("ibvs_rms_px", "IBVS RMS Error [px]"),
    ("ibvs_lin_vel_l2", "IBVS Transl. Speed L2 [m/s]"),
    ("ibvs_ang_vel_l2", "IBVS Rot. Speed L2 [rad/s]"),
    ("ibvs_lin_acc_l2", "IBVS Transl. Acc L2 [m/s^2]"),
    ("ibvs_ang_acc_l2", "IBVS Rot. Acc L2 [rad/s^2]"),
    ("base_lin_vel_l2", "Base Transl. Speed L2 [m/s]"),
    ("base_ang_vel_l2", "Base Rot. Speed L2 [rad/s]"),
    ("base_lin_acc_l2", "Base Transl. Acc L2 [m/s^2]"),
    ("base_ang_acc_l2", "Base Rot. Acc L2 [rad/s^2]"),
]


@dataclass
class RunSeries:
    path: Path
    level: str
    stage: str
    scenario: str
    run_index: int
    t_rel_s: np.ndarray
    metrics: Dict[str, np.ndarray]
    goal_reached: np.ndarray
    run_timed_out: np.ndarray
    motion_start_idx: int

    @property
    def t_motion(self) -> np.ndarray:
        return self.t_rel_s[self.motion_start_idx :] - self.t_rel_s[self.motion_start_idx]


def parse_bool_cell(v: str) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def parse_float_cell(v: str) -> float:
    s = str(v).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def discover_latest_bench(log_root: Path) -> Optional[Path]:
    benches = sorted(log_root.glob("bench_*"), reverse=True)
    return benches[0] if benches else None


def infer_stage_from_path(csv_path: Path) -> str:
    parent = csv_path.parent.name
    return STAGE_DIR_TO_KEY.get(parent, parent)


def find_csv_files(bench_dir: Path) -> list[Path]:
    return sorted(p for p in bench_dir.rglob("*.csv") if p.is_file())


def compute_motion_start(
    t: np.ndarray,
    ibvs_lin_vel_l2: np.ndarray,
    ibvs_ang_vel_l2: np.ndarray,
    ibvs_rms_px: np.ndarray,
    lin_eps: float,
    ang_eps: float,
) -> int:
    if t.size == 0:
        return 0
    c1 = np.isfinite(ibvs_lin_vel_l2) & (ibvs_lin_vel_l2 > lin_eps)
    c2 = np.isfinite(ibvs_ang_vel_l2) & (ibvs_ang_vel_l2 > ang_eps)
    idx = np.where(c1 | c2)[0]
    if idx.size > 0:
        return int(idx[0])
    c3 = np.isfinite(ibvs_rms_px) & (ibvs_rms_px > 0.0)
    idx3 = np.where(c3)[0]
    if idx3.size > 0:
        return int(idx3[0])
    return 0


def load_run(csv_path: Path, lin_eps: float, ang_eps: float) -> Optional[RunSeries]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    t = np.array([parse_float_cell(r.get("t_rel_s", "")) for r in rows], dtype=np.float64)
    run_index = int(parse_float_cell(rows[0].get("run_index", "0")) or 0)
    level = str(rows[0].get("level", "")).strip() or "unknown_level"
    scenario = str(rows[0].get("scenario", "")).strip() or "unknown_scenario"
    stage = str(rows[0].get("stage", "")).strip() or infer_stage_from_path(csv_path)

    metric_arrays: Dict[str, np.ndarray] = {}
    for metric, _ in DEFAULT_METRICS:
        metric_arrays[metric] = np.array([parse_float_cell(r.get(metric, "")) for r in rows], dtype=np.float64)

    metric_arrays["filter_reject_count"] = np.array(
        [parse_float_cell(r.get("filter_reject_count", "")) for r in rows], dtype=np.float64
    )
    metric_arrays["local_rescue_attempts_delta"] = np.array(
        [parse_float_cell(r.get("local_rescue_attempts_delta", "")) for r in rows], dtype=np.float64
    )
    metric_arrays["local_rescue_success_delta"] = np.array(
        [parse_float_cell(r.get("local_rescue_success_delta", "")) for r in rows], dtype=np.float64
    )
    metric_arrays["local_rescue_reject_delta"] = np.array(
        [parse_float_cell(r.get("local_rescue_reject_delta", "")) for r in rows], dtype=np.float64
    )

    goal_reached = np.array([parse_bool_cell(r.get("goal_reached", "false")) for r in rows], dtype=bool)
    run_timed_out = np.array([parse_bool_cell(r.get("run_timed_out", "false")) for r in rows], dtype=bool)

    start_idx = compute_motion_start(
        t=t,
        ibvs_lin_vel_l2=metric_arrays["ibvs_lin_vel_l2"],
        ibvs_ang_vel_l2=metric_arrays["ibvs_ang_vel_l2"],
        ibvs_rms_px=metric_arrays["ibvs_rms_px"],
        lin_eps=lin_eps,
        ang_eps=ang_eps,
    )
    start_idx = max(0, min(start_idx, t.size - 1))

    return RunSeries(
        path=csv_path,
        level=level,
        stage=stage,
        scenario=scenario,
        run_index=run_index,
        t_rel_s=t,
        metrics=metric_arrays,
        goal_reached=goal_reached,
        run_timed_out=run_timed_out,
        motion_start_idx=start_idx,
    )


def safe_nanmean(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    if not np.any(np.isfinite(v)):
        return float("nan")
    return float(np.nanmean(v))


def safe_nanmax(v: np.ndarray) -> float:
    if v.size == 0 or not np.any(np.isfinite(v)):
        return float("nan")
    return float(np.nanmax(v))


def trapz_valid(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapz(y[mask], x[mask]))


def compute_run_summary(run: RunSeries) -> dict[str, Any]:
    t = run.t_motion
    goal_after_start = run.goal_reached[run.motion_start_idx :]
    timeout_after_start = run.run_timed_out[run.motion_start_idx :]
    reached = bool(np.any(goal_after_start))
    timed_out = bool(np.any(timeout_after_start))

    time_to_goal = float("nan")
    if reached:
        idx_local = int(np.where(goal_after_start)[0][0])
        time_to_goal = float(t[idx_local])

    duration = float(t[-1]) if t.size > 0 else float("nan")

    rms = run.metrics["ibvs_rms_px"][run.motion_start_idx :]
    base_lin = run.metrics["base_lin_vel_l2"][run.motion_start_idx :]
    base_ang = run.metrics["base_ang_vel_l2"][run.motion_start_idx :]
    ibvs_lin = run.metrics["ibvs_lin_vel_l2"][run.motion_start_idx :]
    ibvs_ang = run.metrics["ibvs_ang_vel_l2"][run.motion_start_idx :]

    return {
        "csv_file": str(run.path),
        "level": run.level,
        "stage": run.stage,
        "scenario": run.scenario,
        "run_index": run.run_index,
        "samples": int(t.size),
        "motion_start_t_rel_s": float(run.t_rel_s[run.motion_start_idx]),
        "duration_from_motion_start_s": duration,
        "goal_reached": reached,
        "timed_out": timed_out,
        "time_to_goal_s": time_to_goal,
        "ibvs_rms_mean_px": safe_nanmean(rms),
        "ibvs_rms_max_px": safe_nanmax(rms),
        "ibvs_rms_auc_px_s": trapz_valid(rms, t),
        "ibvs_lin_vel_mean_l2": safe_nanmean(ibvs_lin),
        "ibvs_ang_vel_mean_l2": safe_nanmean(ibvs_ang),
        "base_lin_vel_mean_l2": safe_nanmean(base_lin),
        "base_ang_vel_mean_l2": safe_nanmean(base_ang),
        "path_length_base_m": trapz_valid(base_lin, t),
        "base_rotation_integral_rad": trapz_valid(base_ang, t),
        "filter_reject_count_final": int(np.nan_to_num(safe_nanmax(run.metrics["filter_reject_count"]), nan=0.0)),
        "local_rescue_attempts_final": int(
            np.nan_to_num(safe_nanmax(run.metrics["local_rescue_attempts_delta"]), nan=0.0)
        ),
        "local_rescue_success_final": int(
            np.nan_to_num(safe_nanmax(run.metrics["local_rescue_success_delta"]), nan=0.0)
        ),
        "local_rescue_reject_final": int(
            np.nan_to_num(safe_nanmax(run.metrics["local_rescue_reject_delta"]), nan=0.0)
        ),
    }


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_key(run: RunSeries) -> tuple[str, str, str]:
    return (run.level, run.stage, run.scenario)


def sanitize_key(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _finite_minmax(values: np.ndarray, default_min: float, default_max: float) -> tuple[float, float]:
    vv = values[np.isfinite(values)]
    if vv.size == 0:
        return default_min, default_max
    vmin = float(np.min(vv))
    vmax = float(np.max(vv))
    if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
        pad = 1.0 if abs(vmin) < 1e-9 else abs(vmin) * 0.2
        return vmin - pad, vmax + pad
    return vmin, vmax


def _decimate_xy(x: np.ndarray, y: np.ndarray, max_points: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def _write_svg_line_plot(
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 760
    ml, mr, mt, mb = 85, 40, 55, 80
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    x0, y0 = ml, mt

    all_x = []
    all_y = []
    filtered_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, x, y in curves:
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            continue
        xv, yv = x[mask], y[mask]
        xv, yv = _decimate_xy(xv, yv, max_points=1500)
        filtered_curves.append((label, xv, yv))
        all_x.append(xv)
        all_y.append(yv)

    if not filtered_curves:
        with out_path.open("w", encoding="utf-8") as f:
            f.write(
                "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='300'>"
                "<text x='20' y='40' font-size='24'>No finite data for plot</text></svg>"
            )
        return

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    xmin, xmax = _finite_minmax(x_all, 0.0, 1.0)
    ymin, ymax = _finite_minmax(y_all, 0.0, 1.0)

    def sx(x: float) -> float:
        if xmax <= xmin:
            return x0
        return x0 + (x - xmin) * (plot_w / (xmax - xmin))

    def sy(y: float) -> float:
        if ymax <= ymin:
            return y0 + plot_h
        return y0 + plot_h - (y - ymin) * (plot_h / (ymax - ymin))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    lines: list[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='22'>{_svg_escape(title)}</text>")

    lines.append(
        f"<rect x='{x0}' y='{y0}' width='{plot_w}' height='{plot_h}' fill='none' stroke='#333' stroke-width='1.2'/>"
    )

    for i in range(6):
        tx = xmin + (xmax - xmin) * (i / 5.0)
        px = sx(tx)
        lines.append(
            f"<line x1='{px:.2f}' y1='{y0 + plot_h:.2f}' x2='{px:.2f}' y2='{y0 + plot_h + 6:.2f}' stroke='#333'/>"
        )
        lines.append(
            f"<text x='{px:.2f}' y='{y0 + plot_h + 24:.2f}' text-anchor='middle' font-size='12'>{tx:.2f}</text>"
        )
    for i in range(6):
        ty = ymin + (ymax - ymin) * (i / 5.0)
        py = sy(ty)
        lines.append(f"<line x1='{x0 - 6:.2f}' y1='{py:.2f}' x2='{x0:.2f}' y2='{py:.2f}' stroke='#333'/>")
        lines.append(
            f"<text x='{x0 - 10:.2f}' y='{py + 4:.2f}' text-anchor='end' font-size='12'>{ty:.3g}</text>"
        )

    for idx, (label, x, y) in enumerate(filtered_curves):
        color = colors[idx % len(colors)]
        pts = " ".join(f"{sx(float(xx)):.2f},{sy(float(yy)):.2f}" for xx, yy in zip(x, y))
        lines.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.5' opacity='0.72'/>")
        if idx < 12:
            ly = y0 + 16 + idx * 18
            lx = x0 + plot_w - 210
            lines.append(f"<line x1='{lx}' y1='{ly}' x2='{lx + 22}' y2='{ly}' stroke='{color}' stroke-width='2.4'/>")
            lines.append(f"<text x='{lx + 28}' y='{ly + 4}' font-size='12'>{_svg_escape(label)}</text>")

    lines.append(f"<text x='{x0 + plot_w/2:.1f}' y='{height - 22:.1f}' text-anchor='middle' font-size='14'>{_svg_escape(x_label)}</text>")
    lines.append(
        f"<text x='24' y='{y0 + plot_h/2:.1f}' font-size='14' transform='rotate(-90 24,{y0 + plot_h/2:.1f})' text-anchor='middle'>{_svg_escape(y_label)}</text>"
    )
    lines.append("</svg>")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_svg_grouped_bars(
    categories: list[str],
    series: list[tuple[str, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not categories or not series:
        return

    width, height = 1320, 760
    ml, mr, mt, mb = 90, 35, 55, 160
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    x0, y0 = ml, mt

    vals = []
    for _, s in series:
        vals.append(np.nan_to_num(s, nan=0.0))
    allv = np.concatenate(vals) if vals else np.array([1.0], dtype=np.float64)
    _, vmax = _finite_minmax(allv.astype(np.float64), 0.0, 1.0)
    vmax = max(vmax, 1e-9)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    n_cat = len(categories)
    n_ser = len(series)
    group_w = plot_w / max(1, n_cat)
    bar_w = max(3.0, min(42.0, (group_w * 0.82) / max(1, n_ser)))

    lines: list[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='22'>{_svg_escape(title)}</text>")
    lines.append(f"<rect x='{x0}' y='{y0}' width='{plot_w}' height='{plot_h}' fill='none' stroke='#333' stroke-width='1.2'/>")

    for i in range(6):
        yv = vmax * (i / 5.0)
        py = y0 + plot_h - (yv / vmax) * plot_h
        lines.append(f"<line x1='{x0-6:.2f}' y1='{py:.2f}' x2='{x0:.2f}' y2='{py:.2f}' stroke='#333'/>")
        lines.append(f"<text x='{x0-10:.2f}' y='{py+4:.2f}' text-anchor='end' font-size='12'>{yv:.3g}</text>")

    for ci, cat in enumerate(categories):
        gx = x0 + (ci + 0.5) * group_w
        base_left = gx - (n_ser * bar_w) / 2.0
        for si, (name, values) in enumerate(series):
            v = float(np.nan_to_num(values[ci], nan=0.0))
            h = (v / vmax) * plot_h
            bx = base_left + si * bar_w
            by = y0 + plot_h - h
            color = colors[si % len(colors)]
            lines.append(
                f"<rect x='{bx:.2f}' y='{by:.2f}' width='{bar_w*0.9:.2f}' height='{h:.2f}' fill='{color}' opacity='0.85'/>"
            )
        lines.append(
            f"<text x='{gx:.2f}' y='{y0 + plot_h + 18:.2f}' transform='rotate(32 {gx:.2f},{y0 + plot_h + 18:.2f})' text-anchor='start' font-size='11'>{_svg_escape(cat)}</text>"
        )

    for i, (name, _) in enumerate(series):
        lx = x0 + 12 + i * 240
        ly = y0 + 18
        color = colors[i % len(colors)]
        lines.append(f"<rect x='{lx}' y='{ly-10}' width='16' height='10' fill='{color}'/>")
        lines.append(f"<text x='{lx + 22}' y='{ly}' font-size='12'>{_svg_escape(name)}</text>")

    lines.append("</svg>")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_metric_overlay(
    runs: list[RunSeries],
    metric: str,
    y_label: str,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(11, 6))
        for run in runs:
            t = run.t_motion
            y = run.metrics[metric][run.motion_start_idx :]
            if t.size == 0:
                continue
            plt.plot(t, y, alpha=0.45, linewidth=1.1, label=f"run{run.run_index:02d}")
        plt.xlabel("t from motion start [s]")
        plt.ylabel(y_label)
        plt.title(title)
        if len(runs) <= 12:
            plt.legend(loc="best", fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
        return

    curves = []
    for run in runs:
        curves.append((f"run{run.run_index:02d}", run.t_motion, run.metrics[metric][run.motion_start_idx :]))
    _write_svg_line_plot(
        curves=curves,
        title=title,
        x_label="t from motion start [s]",
        y_label=y_label,
        out_path=out_path.with_suffix(".svg"),
    )


def plot_summary_bars(summary_rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return

    labels = [f"run{int(r['run_index']):02d}" for r in summary_rows]
    t_goal = np.array([float(r["time_to_goal_s"]) for r in summary_rows], dtype=np.float64)
    path_len = np.array([float(r["path_length_base_m"]) for r in summary_rows], dtype=np.float64)
    rms_auc = np.array([float(r["ibvs_rms_auc_px_s"]) for r in summary_rows], dtype=np.float64)

    if MATPLOTLIB_AVAILABLE:
        x = np.arange(len(labels))
        w = 0.26
        plt.figure(figsize=(12, 6))
        plt.bar(x - w, np.nan_to_num(t_goal, nan=0.0), width=w, label="time_to_goal [s]")
        plt.bar(x, np.nan_to_num(path_len, nan=0.0), width=w, label="path_length [m]")
        plt.bar(x + w, np.nan_to_num(rms_auc, nan=0.0), width=w, label="rms_auc [px*s]")
        plt.xticks(x, labels, rotation=0)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
        return

    _write_svg_grouped_bars(
        categories=labels,
        series=[
            ("time_to_goal [s]", np.nan_to_num(t_goal, nan=0.0)),
            ("path_length [m]", np.nan_to_num(path_len, nan=0.0)),
            ("rms_auc [px*s]", np.nan_to_num(rms_auc, nan=0.0)),
        ],
        title=title,
        out_path=out_path.with_suffix(".svg"),
    )


def aggregate_group_summaries(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (str(row["level"]), str(row["stage"]), str(row["scenario"]))
        by_group.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (level, stage, scenario), rows in sorted(by_group.items()):
        n = len(rows)
        success = sum(1 for r in rows if bool(r["goal_reached"]))
        timeout = sum(1 for r in rows if bool(r["timed_out"]))
        t_goal = np.array([float(r["time_to_goal_s"]) for r in rows], dtype=np.float64)
        rms_mean = np.array([float(r["ibvs_rms_mean_px"]) for r in rows], dtype=np.float64)
        path_len = np.array([float(r["path_length_base_m"]) for r in rows], dtype=np.float64)
        out.append(
            {
                "level": level,
                "stage": stage,
                "scenario": scenario,
                "runs": n,
                "success_count": success,
                "success_rate": float(success / n) if n > 0 else float("nan"),
                "timeout_count": timeout,
                "time_to_goal_median_s": float(np.nanmedian(t_goal)) if np.any(np.isfinite(t_goal)) else float("nan"),
                "time_to_goal_mean_s": safe_nanmean(t_goal),
                "ibvs_rms_mean_px": safe_nanmean(rms_mean),
                "path_length_mean_m": safe_nanmean(path_len),
            }
        )
    return out


def plot_stage_comparison(group_rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(group_rows) < 2:
        return

    labels = [f"{r['level']} | {r['stage']} | {r['scenario']}" for r in group_rows]
    success = np.array([float(r["success_rate"]) for r in group_rows], dtype=np.float64)
    t_goal = np.array([float(r["time_to_goal_median_s"]) for r in group_rows], dtype=np.float64)
    rms = np.array([float(r["ibvs_rms_mean_px"]) for r in group_rows], dtype=np.float64)

    if MATPLOTLIB_AVAILABLE:
        x = np.arange(len(labels))
        w = 0.28
        plt.figure(figsize=(max(11, len(labels) * 1.1), 6))
        plt.bar(x - w, np.nan_to_num(success, nan=0.0), width=w, label="success_rate")
        plt.bar(x, np.nan_to_num(t_goal, nan=0.0), width=w, label="median_time_to_goal [s]")
        plt.bar(x + w, np.nan_to_num(rms, nan=0.0), width=w, label="mean_ibvs_rms [px]")
        plt.xticks(x, labels, rotation=28, ha="right")
        plt.title("Group Comparison")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=170)
        plt.close()
        return

    _write_svg_grouped_bars(
        categories=labels,
        series=[
            ("success_rate", np.nan_to_num(success, nan=0.0)),
            ("median_time_to_goal [s]", np.nan_to_num(t_goal, nan=0.0)),
            ("mean_ibvs_rms [px]", np.nan_to_num(rms, nan=0.0)),
        ],
        title="Group Comparison",
        out_path=out_path.with_suffix(".svg"),
    )


def write_report(
    out_path: Path,
    bench_dir: Path,
    csv_count: int,
    run_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Benchmark Plot Report")
    lines.append("")
    lines.append(f"- bench_dir: `{bench_dir}`")
    lines.append(f"- csv_files: `{csv_count}`")
    lines.append(f"- loaded_runs: `{len(run_rows)}`")
    lines.append(f"- groups: `{len(group_rows)}`")
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    lines.append("| level | stage | scenario | runs | success_rate | median_time_to_goal_s | mean_ibvs_rms_px |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in group_rows:
        lines.append(
            f"| {r['level']} | {r['stage']} | {r['scenario']} | {r['runs']} | "
            f"{float(r['success_rate']):.3f} | {float(r['time_to_goal_median_s']):.3f} | {float(r['ibvs_rms_mean_px']):.3f} |"
        )
    lines.append("")
    lines.append("Generated files:")
    lines.append("- `run_summary.csv`")
    lines.append("- `group_summary.csv`")
    lines.append(f"- `group_*/*.{ 'png' if MATPLOTLIB_AVAILABLE else 'svg' }`")
    lines.append(f"- `group_comparison.{ 'png' if MATPLOTLIB_AVAILABLE else 'svg' }`")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot benchmark CSV analysis curves.")
    p.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="Benchmark directory (bench_YYYYmmdd_HHMMSS). If omitted, latest under --log-root is used.",
    )
    p.add_argument(
        "--log-root",
        type=Path,
        default=Path("/home/ros_ws/logs/benchmark_csv"),
        help="Root folder containing bench_* directories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <bench_dir>/analysis_plots",
    )
    p.add_argument("--motion-lin-eps", type=float, default=1e-4, help="Threshold for motion-start detection (linear).")
    p.add_argument("--motion-ang-eps", type=float, default=1e-4, help="Threshold for motion-start detection (angular).")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    bench_dir = args.bench_dir
    if bench_dir is None:
        bench_dir = discover_latest_bench(args.log_root)
    if bench_dir is None:
        print(f"[error] no bench_* directory found in {args.log_root}")
        return 2
    if not bench_dir.exists():
        print(f"[error] bench dir not found: {bench_dir}")
        return 2

    out_dir = args.out_dir if args.out_dir is not None else (bench_dir / "analysis_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not MATPLOTLIB_AVAILABLE:
        print("[warn] matplotlib not available. Falling back to SVG output.")

    csv_files = find_csv_files(bench_dir)
    if not csv_files:
        print(f"[error] no csv files found in {bench_dir}")
        return 2

    runs: list[RunSeries] = []
    for p in csv_files:
        run = load_run(csv_path=p, lin_eps=args.motion_lin_eps, ang_eps=args.motion_ang_eps)
        if run is not None:
            runs.append(run)

    if not runs:
        print("[error] could not parse any run CSV.")
        return 2

    runs_by_group: dict[tuple[str, str, str], list[RunSeries]] = {}
    for run in runs:
        runs_by_group.setdefault(group_key(run), []).append(run)
    for key in runs_by_group:
        runs_by_group[key].sort(key=lambda r: r.run_index)

    run_summary_rows = [compute_run_summary(r) for r in runs]
    write_csv(run_summary_rows, out_dir / "run_summary.csv")

    for (level, stage, scenario), group_runs in sorted(runs_by_group.items()):
        g_name = sanitize_key(f"{level}__{stage}__{scenario}")
        g_dir = out_dir / f"group_{g_name}"
        g_dir.mkdir(parents=True, exist_ok=True)

        for metric, y_label in DEFAULT_METRICS:
            plot_metric_overlay(
                runs=group_runs,
                metric=metric,
                y_label=y_label,
                out_path=g_dir / f"{metric}.png",
                title=f"{level} | {stage} | {scenario} | {metric}",
            )

        plot_metric_overlay(
            runs=group_runs,
            metric="filter_reject_count",
            y_label="Filter Reject Count",
            out_path=g_dir / "filter_reject_count.png",
            title=f"{level} | {stage} | {scenario} | filter_reject_count",
        )
        plot_metric_overlay(
            runs=group_runs,
            metric="local_rescue_attempts_delta",
            y_label="Local Rescue Attempts",
            out_path=g_dir / "local_rescue_attempts.png",
            title=f"{level} | {stage} | {scenario} | local_rescue_attempts",
        )
        plot_metric_overlay(
            runs=group_runs,
            metric="local_rescue_success_delta",
            y_label="Local Rescue Success",
            out_path=g_dir / "local_rescue_success.png",
            title=f"{level} | {stage} | {scenario} | local_rescue_success",
        )
        plot_metric_overlay(
            runs=group_runs,
            metric="local_rescue_reject_delta",
            y_label="Local Rescue Reject",
            out_path=g_dir / "local_rescue_reject.png",
            title=f"{level} | {stage} | {scenario} | local_rescue_reject",
        )

        group_run_rows = [r for r in run_summary_rows if r["level"] == level and r["stage"] == stage and r["scenario"] == scenario]
        plot_summary_bars(
            summary_rows=group_run_rows,
            out_path=g_dir / "run_summary_bars.png",
            title=f"{level} | {stage} | {scenario} | run summary",
        )

    group_summary_rows = aggregate_group_summaries(run_summary_rows)
    write_csv(group_summary_rows, out_dir / "group_summary.csv")
    plot_stage_comparison(group_summary_rows, out_dir / "group_comparison.png")
    write_report(
        out_path=out_dir / "report.md",
        bench_dir=bench_dir,
        csv_count=len(csv_files),
        run_rows=run_summary_rows,
        group_rows=group_summary_rows,
    )

    print(f"[ok] loaded runs: {len(runs)}")
    print(f"[ok] groups: {len(group_summary_rows)}")
    print(f"[ok] output: {out_dir}")
    print(f"[ok] run summary: {out_dir / 'run_summary.csv'}")
    print(f"[ok] group summary: {out_dir / 'group_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
