#!/usr/bin/env python3
"""Identify likely IBVS pipeline bottlenecks from rosbag2 sqlite data.

Supports:
- direct bag folder with *.db3
- benchmark run folder with mode subdirs (off/shadow/active)

Detection is based on throughput drops between pipeline stages:
camera -> keypoints -> matches -> filtered_features -> cmd_vel
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import struct
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


PIPELINE_STAGES: List[Tuple[str, List[str]]] = [
    (
        "camera",
        [
            "/camera/camera/color/image_raw",
            "/camera/camera/color/image_raw/compressed",
        ],
    ),
    ("keypoints", ["/ibvs/keypoints"]),
    ("matches", ["/ibvs/matches"]),
    ("filtered", ["/ibvs/filtered_features"]),
    ("twist_cmd", ["/cartesian_twist_passthrough_controller/cmd_vel"]),
]
PIPELINE_STAGE_NAMES = [name for name, _ in PIPELINE_STAGES]

COUNTER_TOPICS: Dict[str, str] = {
    "update_count": "/ibvs/filter/update_count",
    "update_success_count": "/ibvs/filter/update_success_count",
    "active_count": "/ibvs/filter/active_count",
    "local_rescue_attempts": "/ibvs/matching/local_rescue_attempts",
    "local_rescue_success": "/ibvs/matching/local_rescue_success",
    "local_rescue_reject": "/ibvs/matching/local_rescue_reject",
}

FLOAT_TOPICS: Dict[str, str] = {
    "uncertainty": "/ibvs/filter/uncertainty",
}

NODE_BY_LINK: Dict[Tuple[str, str], str] = {
    ("camera", "keypoints"): "ibvs_perception/keypoint_node",
    ("keypoints", "matches"): "ibvs_matching/descriptor_matcher_node",
    ("matches", "filtered"): "ibvs_filter/filter_node",
    ("filtered", "twist_cmd"): "ibvs_control/ibvs_twist_controller_node",
}

LIVE_RESCUE_TOPICS: Dict[str, str] = {
    "local_rescue_attempts": "/ibvs/matching/local_rescue_attempts",
    "local_rescue_success": "/ibvs/matching/local_rescue_success",
    "local_rescue_reject": "/ibvs/matching/local_rescue_reject",
    "local_rescue_debug": "/ibvs/matching/local_rescue_debug",
}


@dataclass
class TopicStats:
    stage: str
    topic: str
    topic_type: Optional[str]
    count: int
    rate_hz: float
    dt_p50_ms: Optional[float]
    dt_p95_ms: Optional[float]
    jitter_cv: Optional[float]
    publisher_nodes: List[str]


@dataclass
class LinkStats:
    upstream_stage: str
    downstream_stage: str
    ratio: Optional[float]


@dataclass
class AnalysisResult:
    label: str
    source: str
    duration_s: float
    message_count: int
    stages: List[TopicStats]
    links: List[LinkStats]
    bottleneck_link: Optional[LinkStats]
    likely_node: Optional[str]
    likely_node_publishers: List[str]
    counters: Dict[str, Optional[int]]
    floats: Dict[str, Optional[float]]
    derived: Dict[str, Optional[float]]
    hints: List[str]


def percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    p = max(0.0, min(100.0, pct)) / 100.0
    pos = (len(sorted_values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def decode_uint32(blob: bytes) -> int:
    return int(struct.unpack_from("<I", blob, 4)[0])


def decode_float32(blob: bytes) -> float:
    return float(struct.unpack_from("<f", blob, 4)[0])


def safe_div(num: float, den: float) -> Optional[float]:
    if den <= 1e-12:
        return None
    return num / den


def fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def enabled_stage_defs(disabled_stages: Optional[List[str]]) -> List[Tuple[str, List[str]]]:
    disabled = {s.strip() for s in (disabled_stages or []) if str(s).strip()}
    return [(name, cands) for name, cands in PIPELINE_STAGES if name not in disabled]


def resolve_inputs(path: str, mode: str) -> Dict[str, str]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise RuntimeError(f"path not found: {path}")

    if os.path.isfile(path) and path.endswith(".db3"):
        return {"single": path}

    if not os.path.isdir(path):
        raise RuntimeError(f"not a directory: {path}")

    direct_db = sorted(glob.glob(os.path.join(path, "*.db3")))
    if direct_db:
        return {"single": direct_db[0]}

    if mode != "auto":
        mode_db = sorted(glob.glob(os.path.join(path, mode, "*.db3")))
        if not mode_db:
            raise RuntimeError(f"no db3 found for mode '{mode}' in: {path}")
        return {mode: mode_db[0]}

    out: Dict[str, str] = {}
    for m in ("off", "shadow", "active"):
        mode_db = sorted(glob.glob(os.path.join(path, m, "*.db3")))
        if mode_db:
            out[m] = mode_db[0]

    if out:
        return out

    recursive_db = sorted(glob.glob(os.path.join(path, "**", "*.db3"), recursive=True))
    if len(recursive_db) == 1:
        return {"single": recursive_db[0]}

    if recursive_db:
        raise RuntimeError(
            "multiple db3 files found. Pass --mode or a more specific path."
        )
    raise RuntimeError(f"no db3 files found under: {path}")


def topic_name_to_id(conn: sqlite3.Connection) -> Dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM topics")
    return {name: int(tid) for tid, name in cur.fetchall()}


def fetch_timestamps(conn: sqlite3.Connection, topic_id: int) -> List[int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp ASC",
        (topic_id,),
    )
    return [int(row[0]) for row in cur.fetchall()]


def fetch_first_last_blob(conn: sqlite3.Connection, topic_id: int) -> Tuple[Optional[bytes], Optional[bytes]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
        (topic_id,),
    )
    first_row = cur.fetchone()
    cur.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1",
        (topic_id,),
    )
    last_row = cur.fetchone()
    first_blob = first_row[0] if first_row else None
    last_blob = last_row[0] if last_row else None
    return first_blob, last_blob


def mean_of_topic(conn: sqlite3.Connection, topic_id: int, decode_fn) -> Optional[float]:
    cur = conn.cursor()
    cur.execute("SELECT data FROM messages WHERE topic_id=?", (topic_id,))
    values = [float(decode_fn(row[0])) for row in cur.fetchall()]
    if not values:
        return None
    return float(sum(values) / len(values))


def choose_stage_topic(name_to_id: Dict[str, int], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in name_to_id:
            return c
    return None


def build_topic_stats(
    conn: sqlite3.Connection,
    stage: str,
    topic: str,
    topic_id: int,
    duration_s: float,
    topic_type: Optional[str] = None,
    publisher_nodes: Optional[List[str]] = None,
) -> TopicStats:
    ts = fetch_timestamps(conn, topic_id)
    count = len(ts)
    rate_hz = float(count / duration_s) if duration_s > 1e-9 else 0.0

    dt_p50_ms: Optional[float] = None
    dt_p95_ms: Optional[float] = None
    jitter_cv: Optional[float] = None
    if count >= 2:
        dt_s = [(ts[i] - ts[i - 1]) * 1e-9 for i in range(1, count)]
        dt_sorted = sorted(dt_s)
        dt_p50 = percentile(dt_sorted, 50.0)
        dt_p95 = percentile(dt_sorted, 95.0)
        dt_p50_ms = None if dt_p50 is None else dt_p50 * 1000.0
        dt_p95_ms = None if dt_p95 is None else dt_p95 * 1000.0
        mu = sum(dt_s) / len(dt_s)
        if mu > 1e-12:
            var = sum((x - mu) ** 2 for x in dt_s) / len(dt_s)
            jitter_cv = (var ** 0.5) / mu

    return TopicStats(
        stage=stage,
        topic=topic,
        topic_type=topic_type,
        count=count,
        rate_hz=rate_hz,
        dt_p50_ms=dt_p50_ms,
        dt_p95_ms=dt_p95_ms,
        jitter_cv=jitter_cv,
        publisher_nodes=publisher_nodes or [],
    )


def select_bottleneck_link(
    links: List[LinkStats],
    stage_map: Dict[str, TopicStats],
    min_stage_rate_hz: float,
) -> Optional[LinkStats]:
    bottleneck_link: Optional[LinkStats] = None
    best_loss = 0.0
    for link in links:
        up = stage_map.get(link.upstream_stage)
        if link.ratio is None or up is None:
            continue
        if up.rate_hz < min_stage_rate_hz:
            continue
        ratio = link.ratio
        if link.upstream_stage == "matches" and link.downstream_stage == "filtered":
            # Filter can publish faster than matches due to predict timer.
            # Only penalize clear under-performance.
            loss = max(0.0, (0.8 - ratio) / 0.8)
        else:
            loss = max(0.0, 1.0 - ratio)
        if loss > best_loss:
            best_loss = loss
            bottleneck_link = link

    if best_loss < 0.05:
        return None
    return bottleneck_link


def analyze_db(
    db_path: str,
    label: str,
    expected_camera_fps: float,
    min_stage_rate_hz: float,
    disabled_stages: Optional[List[str]] = None,
) -> AnalysisResult:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages")
    message_count, t_min, t_max = cur.fetchone()
    message_count = int(message_count or 0)
    duration_s = 0.0
    if t_min is not None and t_max is not None and int(t_max) >= int(t_min):
        duration_s = (int(t_max) - int(t_min)) * 1e-9

    name_to_id = topic_name_to_id(conn)

    stage_defs = enabled_stage_defs(disabled_stages)

    stages: List[TopicStats] = []
    stage_map: Dict[str, TopicStats] = {}
    for stage_name, candidates in stage_defs:
        topic = choose_stage_topic(name_to_id, candidates)
        if topic is None:
            continue
        stat = build_topic_stats(
            conn=conn,
            stage=stage_name,
            topic=topic,
            topic_id=name_to_id[topic],
            duration_s=duration_s,
            topic_type=None,
            publisher_nodes=[],
        )
        stages.append(stat)
        stage_map[stage_name] = stat

    links: List[LinkStats] = []
    ordered_stage_names = [name for name, _ in stage_defs]
    for i in range(1, len(ordered_stage_names)):
        up_name = ordered_stage_names[i - 1]
        dn_name = ordered_stage_names[i]
        up = stage_map.get(up_name)
        dn = stage_map.get(dn_name)
        if up is None or dn is None:
            continue
        ratio = safe_div(dn.rate_hz, up.rate_hz)
        links.append(
            LinkStats(
                upstream_stage=up_name,
                downstream_stage=dn_name,
                ratio=ratio,
            )
        )

    bottleneck_link = select_bottleneck_link(
        links=links,
        stage_map=stage_map,
        min_stage_rate_hz=min_stage_rate_hz,
    )

    counters: Dict[str, Optional[int]] = {}
    for k, topic in COUNTER_TOPICS.items():
        tid = name_to_id.get(topic)
        if tid is None:
            counters[k] = None
            continue
        first_blob, last_blob = fetch_first_last_blob(conn, tid)
        if first_blob is None or last_blob is None:
            counters[k] = None
            continue
        last_val = decode_uint32(last_blob)
        if k in ("update_count", "update_success_count", "local_rescue_attempts", "local_rescue_success", "local_rescue_reject"):
            first_val = decode_uint32(first_blob)
            counters[k] = int(last_val - first_val)
        else:
            counters[k] = int(last_val)

    floats: Dict[str, Optional[float]] = {}
    for k, topic in FLOAT_TOPICS.items():
        tid = name_to_id.get(topic)
        if tid is None:
            floats[k] = None
            continue
        floats[k] = mean_of_topic(conn, tid, decode_float32)

    derived: Dict[str, Optional[float]] = {}
    matches_count = stage_map["matches"].count if "matches" in stage_map else None
    update_success = counters.get("update_success_count")
    if matches_count is not None and update_success is not None and matches_count > 0:
        derived["update_success_per_match"] = update_success / float(matches_count)
    else:
        derived["update_success_per_match"] = None

    lr_att = counters.get("local_rescue_attempts")
    lr_suc = counters.get("local_rescue_success")
    lr_rej = counters.get("local_rescue_reject")
    if lr_att is not None and lr_suc is not None and lr_att > 0:
        derived["local_rescue_success_rate"] = lr_suc / float(lr_att)
    else:
        derived["local_rescue_success_rate"] = None
    if lr_att is not None and lr_rej is not None and lr_att > 0:
        derived["local_rescue_reject_rate"] = lr_rej / float(lr_att)
    else:
        derived["local_rescue_reject_rate"] = None

    if bottleneck_link is None:
        uspm = derived.get("update_success_per_match")
        if uspm is not None and uspm < 0.65:
            bottleneck_link = LinkStats(
                upstream_stage="matches",
                downstream_stage="filtered",
                ratio=uspm,
            )

    hints: List[str] = []
    cam = stage_map.get("camera")
    kp = stage_map.get("keypoints")
    mt = stage_map.get("matches")
    flt = stage_map.get("filtered")
    tw = stage_map.get("twist_cmd")

    if cam is not None and expected_camera_fps > 1e-9 and cam.rate_hz < 0.7 * expected_camera_fps:
        hints.append(
            f"Kamera-Input ist niedrig ({cam.rate_hz:.2f} Hz < 70% von erwartet {expected_camera_fps:.1f} Hz)."
        )

    if bottleneck_link is not None:
        key = f"{bottleneck_link.upstream_stage}->{bottleneck_link.downstream_stage}"
        if key == "camera->keypoints":
            hints.append("Bottleneck wahrscheinlich in Detektion/Perception (Keypoint-Node).")
        elif key == "keypoints->matches":
            hints.append("Bottleneck wahrscheinlich im Descriptor-Matching.")
        elif key == "matches->filtered":
            hints.append("Bottleneck wahrscheinlich im Filter-Update (Gating/Geometrie/Relokalisierung).")
        elif key == "filtered->twist_cmd":
            hints.append("Bottleneck wahrscheinlich im Controller-Gating (z. B. min_matches/init_done/enable_motion).")

    update_count = counters.get("update_count")
    if update_count is not None and update_success is not None and update_count > 0:
        sr = update_success / float(update_count)
        if sr < 0.6:
            hints.append(
                f"Filter-Update-Erfolgsrate niedrig ({sr:.2%}); Gate/Noise/Geometrie prüfen."
            )

    if lr_att is not None and lr_suc is not None and lr_att > 0:
        rescue_sr = lr_suc / float(lr_att)
        if rescue_sr < 0.4:
            hints.append(
                f"Local-Rescue-Erfolg niedrig ({rescue_sr:.2%}); adaptive gates / sim thresholds prüfen."
            )
    if lr_att is not None and lr_rej is not None and lr_att > 0:
        reject_ratio = lr_rej / float(lr_att)
        if reject_ratio > 0.5:
            hints.append(
                f"Local-Rescue-Reject-Anteil hoch ({reject_ratio:.2%}); Suchradius/Score-Grenzen prüfen."
            )

    if flt is not None and mt is not None and mt.rate_hz > 1e-9:
        ratio = flt.rate_hz / mt.rate_hz
        if ratio < 0.6:
            hints.append("Filter publiziert deutlich seltener als Matches; Update-Pfad priorisiert prüfen.")
    uspm = derived.get("update_success_per_match")
    if uspm is not None and uspm < 0.7:
        hints.append(
            f"Nur {uspm:.2f} erfolgreiche Filter-Updates pro Match im Mittel; "
            "Filter-Selektion/Gating wahrscheinlich limitierend."
        )

    if tw is not None and flt is not None and flt.rate_hz > 1e-9:
        ratio = tw.rate_hz / flt.rate_hz
        if ratio < 0.6:
            hints.append(
                "Twist-Publisher deutlich langsamer als Filter-Output; Controller-Bedingungen prüfen."
            )

    likely_node = None
    likely_node_publishers: List[str] = []
    if bottleneck_link is not None:
        likely_node = NODE_BY_LINK.get(
            (bottleneck_link.upstream_stage, bottleneck_link.downstream_stage)
        )
        dn_stage = stage_map.get(bottleneck_link.downstream_stage)
        if dn_stage is not None:
            likely_node_publishers = list(dn_stage.publisher_nodes)

    conn.close()

    return AnalysisResult(
        label=label,
        source=os.path.abspath(db_path),
        duration_s=float(duration_s),
        message_count=message_count,
        stages=stages,
        links=links,
        bottleneck_link=bottleneck_link,
        likely_node=likely_node,
        likely_node_publishers=likely_node_publishers,
        counters=counters,
        floats=floats,
        derived=derived,
        hints=hints,
    )


def _compute_dt_stats_from_times(times: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(times) < 2:
        return None, None, None
    dt_s = [times[i] - times[i - 1] for i in range(1, len(times))]
    dt_sorted = sorted(dt_s)
    dt_p50 = percentile(dt_sorted, 50.0)
    dt_p95 = percentile(dt_sorted, 95.0)
    dt_p50_ms = None if dt_p50 is None else dt_p50 * 1000.0
    dt_p95_ms = None if dt_p95 is None else dt_p95 * 1000.0
    mu = sum(dt_s) / len(dt_s)
    if mu <= 1e-12:
        return dt_p50_ms, dt_p95_ms, None
    var = sum((x - mu) ** 2 for x in dt_s) / len(dt_s)
    jitter_cv = (var ** 0.5) / mu
    return dt_p50_ms, dt_p95_ms, jitter_cv


def _endpoint_to_name(ep) -> str:
    ns = ep.node_namespace or "/"
    if not ns.startswith("/"):
        ns = "/" + ns
    if ns.endswith("/"):
        return f"{ns}{ep.node_name}"
    return f"{ns}/{ep.node_name}"


def _pick_topic_for_stage(
    topics_and_types: Dict[str, List[str]],
    candidates: List[str],
    override_topic: str = "",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (topic_name, topic_type, warning)."""
    if override_topic:
        types = topics_and_types.get(override_topic, [])
        if types:
            return override_topic, types[0], None
        return None, None, f"Override topic nicht gefunden: {override_topic}"

    for cand in candidates:
        ttypes = topics_and_types.get(cand, [])
        if ttypes:
            return cand, ttypes[0], None

    suffix_hits: List[str] = []
    for topic_name in topics_and_types.keys():
        for cand in candidates:
            if topic_name.endswith(cand):
                suffix_hits.append(topic_name)
                break
    suffix_hits = sorted(set(suffix_hits))
    if len(suffix_hits) == 1:
        name = suffix_hits[0]
        return name, topics_and_types[name][0], None
    if len(suffix_hits) > 1:
        return (
            None,
            None,
            f"Mehrere Kandidaten via Suffix-Match gefunden: {', '.join(suffix_hits)}",
        )
    return None, None, None


def _build_links_from_stage_map(stage_defs: List[Tuple[str, List[str]]], stage_map: Dict[str, TopicStats]) -> List[LinkStats]:
    links: List[LinkStats] = []
    ordered_stage_names = [name for name, _ in stage_defs]
    for i in range(1, len(ordered_stage_names)):
        up_name = ordered_stage_names[i - 1]
        dn_name = ordered_stage_names[i]
        up = stage_map.get(up_name)
        dn = stage_map.get(dn_name)
        if up is None or dn is None:
            continue
        links.append(
            LinkStats(
                upstream_stage=up_name,
                downstream_stage=dn_name,
                ratio=safe_div(dn.rate_hz, up.rate_hz),
            )
        )
    return links


def _render_live_table(
    elapsed_s: float,
    stages: List[TopicStats],
    links: List[LinkStats],
    bottleneck_link: Optional[LinkStats],
    rescue_derived: Dict[str, Optional[float]],
):
    print("\033[2J\033[H", end="")
    print(f"IBVS Bottleneck Monitor (live) | elapsed: {elapsed_s:.1f}s")
    print("")
    print("stage         rate_hz   count    dt_p50_ms   dt_p95_ms   jitter_cv   topic")
    for s in stages:
        print(
            f"{s.stage:<12} {s.rate_hz:>7.2f} {s.count:>8d} "
            f"{fmt(s.dt_p50_ms, 2):>10} {fmt(s.dt_p95_ms, 2):>10} {fmt(s.jitter_cv, 3):>10}   {s.topic}"
        )
    print("")
    print("link ratios (downstream/upstream)")
    if not links:
        print("n/a")
    for link in links:
        mark = ""
        if bottleneck_link is not None and (
            link.upstream_stage == bottleneck_link.upstream_stage
            and link.downstream_stage == bottleneck_link.downstream_stage
        ):
            mark = "  <-- bottleneck"
        print(
            f"{link.upstream_stage:>10} -> {link.downstream_stage:<10}: {fmt(link.ratio, 3)}{mark}"
        )
    print("")
    print("local rescue")
    print(f"debug_rate_hz:           {fmt(rescue_derived.get('local_rescue_debug_rate_hz'), 3)}")
    print(f"attempts_per_sec:        {fmt(rescue_derived.get('local_rescue_attempts_per_sec'), 3)}")
    print(f"success_per_sec:         {fmt(rescue_derived.get('local_rescue_success_per_sec'), 3)}")
    print(f"reject_per_sec:          {fmt(rescue_derived.get('local_rescue_reject_per_sec'), 3)}")
    print(f"success_rate:            {fmt(rescue_derived.get('local_rescue_success_rate'), 3)}")
    print("")
    print("Ctrl+C to stop (or wait for duration).", flush=True)


def analyze_live(
    duration_s: float,
    expected_camera_fps: float,
    min_stage_rate_hz: float,
    live_topic_overrides: Optional[Dict[str, str]] = None,
    discovery_wait_s: float = 3.0,
    disabled_stages: Optional[List[str]] = None,
    live_refresh_s: float = 1.0,
    rescue_topic_overrides: Optional[Dict[str, str]] = None,
) -> AnalysisResult:
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(
            "Live mode benötigt ROS 2 Python-Umgebung (rclpy + rosidl_runtime_py). "
            "Bitte im gesourcten ROS-Environment ausführen."
        ) from exc

    stage_defs = enabled_stage_defs(disabled_stages)
    if not stage_defs:
        raise RuntimeError("Alle Pipeline-Stages sind deaktiviert.")

    rclpy.init()
    node = rclpy.create_node("ibvs_bottleneck_live_monitor")

    try:
        live_topic_overrides = live_topic_overrides or {}
        rescue_topic_overrides = rescue_topic_overrides or {}

        stage_to_topic: Dict[str, str] = {}
        stage_to_type: Dict[str, str] = {}
        stage_to_pub_nodes: Dict[str, List[str]] = {}
        stage_match_warnings: List[str] = []

        rescue_to_topic: Dict[str, str] = {}
        rescue_to_type: Dict[str, str] = {}
        rescue_match_warnings: List[str] = []

        best_topic_count = -1
        discovery_topics_seen = 0
        deadline = time.monotonic() + max(0.0, float(discovery_wait_s))

        while True:
            topics_and_types = dict(node.get_topic_names_and_types())
            discovery_topics_seen = max(discovery_topics_seen, len(topics_and_types))

            cur_stage_to_topic: Dict[str, str] = {}
            cur_stage_to_type: Dict[str, str] = {}
            cur_stage_to_pub_nodes: Dict[str, List[str]] = {}
            cur_stage_match_warnings: List[str] = []
            for stage_name, candidates in stage_defs:
                override_topic = live_topic_overrides.get(stage_name, "")
                topic_name, topic_type, warn = _pick_topic_for_stage(
                    topics_and_types=topics_and_types,
                    candidates=candidates,
                    override_topic=override_topic,
                )
                if warn:
                    cur_stage_match_warnings.append(f"{stage_name}: {warn}")
                if topic_name is None or topic_type is None:
                    continue
                cur_stage_to_topic[stage_name] = topic_name
                cur_stage_to_type[stage_name] = topic_type
                publishers = node.get_publishers_info_by_topic(topic_name)
                cur_stage_to_pub_nodes[stage_name] = sorted(
                    {_endpoint_to_name(ep) for ep in publishers}
                )

            cur_rescue_to_topic: Dict[str, str] = {}
            cur_rescue_to_type: Dict[str, str] = {}
            cur_rescue_match_warnings: List[str] = []
            for rescue_key, default_topic in LIVE_RESCUE_TOPICS.items():
                override_topic = rescue_topic_overrides.get(rescue_key, "")
                topic_name, topic_type, warn = _pick_topic_for_stage(
                    topics_and_types=topics_and_types,
                    candidates=[default_topic],
                    override_topic=override_topic,
                )
                if warn:
                    cur_rescue_match_warnings.append(f"{rescue_key}: {warn}")
                if topic_name is None or topic_type is None:
                    continue
                cur_rescue_to_topic[rescue_key] = topic_name
                cur_rescue_to_type[rescue_key] = topic_type

            stage_score = len(cur_stage_to_topic)
            rescue_score = len(cur_rescue_to_topic)
            if (stage_score > best_topic_count) or (
                stage_score == best_topic_count and rescue_score > len(rescue_to_topic)
            ):
                best_topic_count = stage_score
                stage_to_topic = cur_stage_to_topic
                stage_to_type = cur_stage_to_type
                stage_to_pub_nodes = cur_stage_to_pub_nodes
                stage_match_warnings = cur_stage_match_warnings
                rescue_to_topic = cur_rescue_to_topic
                rescue_to_type = cur_rescue_to_type
                rescue_match_warnings = cur_rescue_match_warnings

            if len(stage_to_topic) >= len(stage_defs):
                break
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)

        recv_times: Dict[str, List[float]] = {k: [] for k in stage_to_topic.keys()}
        rescue_debug_times: List[float] = []
        rescue_counter_state = {
            "local_rescue_attempts": {"first": None, "latest": None},
            "local_rescue_success": {"first": None, "latest": None},
            "local_rescue_reject": {"first": None, "latest": None},
        }

        subscriptions = []
        for stage_name, topic in stage_to_topic.items():
            topic_type = stage_to_type[stage_name]
            msg_cls = get_message(topic_type)

            def _mk_stage_cb(_stage: str):
                def _cb(_msg):
                    recv_times[_stage].append(time.monotonic())

                return _cb

            sub = node.create_subscription(
                msg_cls,
                topic,
                _mk_stage_cb(stage_name),
                qos_profile_sensor_data,
            )
            subscriptions.append(sub)

        for rescue_key, topic in rescue_to_topic.items():
            topic_type = rescue_to_type[rescue_key]
            msg_cls = get_message(topic_type)
            if rescue_key == "local_rescue_debug":
                sub = node.create_subscription(
                    msg_cls,
                    topic,
                    lambda _msg: rescue_debug_times.append(time.monotonic()),
                    qos_profile_sensor_data,
                )
                subscriptions.append(sub)
                continue

            def _mk_counter_cb(_key: str):
                def _cb(_msg):
                    val = int(getattr(_msg, "data", 0))
                    st = rescue_counter_state[_key]
                    if st["first"] is None:
                        st["first"] = val
                    st["latest"] = val

                return _cb

            sub = node.create_subscription(
                msg_cls,
                topic,
                _mk_counter_cb(rescue_key),
                qos_profile_sensor_data,
            )
            subscriptions.append(sub)

        def _build_stage_snapshot(elapsed: float) -> Tuple[List[TopicStats], Dict[str, TopicStats]]:
            stages_local: List[TopicStats] = []
            stage_map_local: Dict[str, TopicStats] = {}
            for stage_name, _ in stage_defs:
                topic = stage_to_topic.get(stage_name)
                if not topic:
                    continue
                ts = recv_times.get(stage_name, [])
                dt_p50_ms, dt_p95_ms, jitter_cv = _compute_dt_stats_from_times(ts)
                stat = TopicStats(
                    stage=stage_name,
                    topic=topic,
                    topic_type=stage_to_type.get(stage_name),
                    count=len(ts),
                    rate_hz=(len(ts) / max(1e-6, elapsed)),
                    dt_p50_ms=dt_p50_ms,
                    dt_p95_ms=dt_p95_ms,
                    jitter_cv=jitter_cv,
                    publisher_nodes=stage_to_pub_nodes.get(stage_name, []),
                )
                stages_local.append(stat)
                stage_map_local[stage_name] = stat
            return stages_local, stage_map_local

        def _build_rescue_snapshot(elapsed: float) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[float]]]:
            counters_local: Dict[str, Optional[int]] = {}
            for k in ("local_rescue_attempts", "local_rescue_success", "local_rescue_reject"):
                st = rescue_counter_state[k]
                if st["first"] is None or st["latest"] is None:
                    counters_local[k] = None
                else:
                    counters_local[k] = int(st["latest"] - st["first"])
            counters_local["local_rescue_debug_frames"] = int(len(rescue_debug_times))

            derived_local: Dict[str, Optional[float]] = {}
            att = counters_local.get("local_rescue_attempts")
            suc = counters_local.get("local_rescue_success")
            rej = counters_local.get("local_rescue_reject")
            if att is not None:
                derived_local["local_rescue_attempts_per_sec"] = att / max(1e-6, elapsed)
            else:
                derived_local["local_rescue_attempts_per_sec"] = None
            if suc is not None:
                derived_local["local_rescue_success_per_sec"] = suc / max(1e-6, elapsed)
            else:
                derived_local["local_rescue_success_per_sec"] = None
            if rej is not None:
                derived_local["local_rescue_reject_per_sec"] = rej / max(1e-6, elapsed)
            else:
                derived_local["local_rescue_reject_per_sec"] = None
            if att is not None and att > 0 and suc is not None:
                derived_local["local_rescue_success_rate"] = suc / float(att)
            else:
                derived_local["local_rescue_success_rate"] = None
            derived_local["local_rescue_debug_rate_hz"] = len(rescue_debug_times) / max(1e-6, elapsed)
            return counters_local, derived_local

        t_start = time.monotonic()
        t_end = t_start + max(0.1, duration_s)
        next_refresh = t_start + max(0.1, live_refresh_s)
        try:
            while time.monotonic() < t_end:
                rclpy.spin_once(node, timeout_sec=0.1)
                now = time.monotonic()
                if live_refresh_s > 0.0 and now >= next_refresh:
                    elapsed = max(1e-6, now - t_start)
                    stages_now, stage_map_now = _build_stage_snapshot(elapsed)
                    links_now = _build_links_from_stage_map(stage_defs, stage_map_now)
                    bottleneck_now = select_bottleneck_link(
                        links=links_now,
                        stage_map=stage_map_now,
                        min_stage_rate_hz=min_stage_rate_hz,
                    )
                    _, rescue_derived_now = _build_rescue_snapshot(elapsed)
                    _render_live_table(
                        elapsed_s=elapsed,
                        stages=stages_now,
                        links=links_now,
                        bottleneck_link=bottleneck_now,
                        rescue_derived=rescue_derived_now,
                    )
                    next_refresh = now + max(0.1, live_refresh_s)
        except KeyboardInterrupt:
            pass

        t_stop = time.monotonic()
        obs_duration_s = max(1e-6, t_stop - t_start)

        stages, stage_map = _build_stage_snapshot(obs_duration_s)
        links = _build_links_from_stage_map(stage_defs, stage_map)
        bottleneck_link = select_bottleneck_link(
            links=links,
            stage_map=stage_map,
            min_stage_rate_hz=min_stage_rate_hz,
        )

        likely_node = None
        likely_node_publishers: List[str] = []
        if bottleneck_link is not None:
            likely_node = NODE_BY_LINK.get(
                (bottleneck_link.upstream_stage, bottleneck_link.downstream_stage)
            )
            dn_stage = stage_map.get(bottleneck_link.downstream_stage)
            if dn_stage is not None:
                likely_node_publishers = list(dn_stage.publisher_nodes)

        counters, rescue_derived = _build_rescue_snapshot(obs_duration_s)

        hints: List[str] = []
        cam = stage_map.get("camera")
        if cam is not None and expected_camera_fps > 1e-9 and cam.rate_hz < 0.7 * expected_camera_fps:
            hints.append(
                f"Kamera-Input ist niedrig ({cam.rate_hz:.2f} Hz < 70% von erwartet {expected_camera_fps:.1f} Hz)."
            )
        if bottleneck_link is not None:
            key = f"{bottleneck_link.upstream_stage}->{bottleneck_link.downstream_stage}"
            if key == "camera->keypoints":
                hints.append("Perception/Detektion ist wahrscheinlich limitierend.")
            elif key == "keypoints->matches":
                hints.append("Descriptor-Matcher ist wahrscheinlich limitierend.")
            elif key == "matches->filtered":
                hints.append("Filter-Node ist wahrscheinlich limitierend.")
            elif key == "filtered->twist_cmd":
                hints.append("IBVS-Twist-Controller ist wahrscheinlich limitierend.")
        else:
            hints.append("Kein klarer Rate-Bottleneck in der Beobachtungszeit erkannt.")
        if stage_match_warnings:
            hints.extend(stage_match_warnings)
        if rescue_match_warnings:
            hints.extend(rescue_match_warnings)
        hints.append(f"Discovery topics seen: {discovery_topics_seen}")
        if not stages:
            hints.append(
                "Keine Pipeline-Topics erkannt. Prüfe ROS_DOMAIN_ID, laufende Topics "
                "(`ros2 topic list`) oder nutze --*-topic Overrides."
            )

        if live_refresh_s > 0.0:
            print("")

        message_count = sum(s.count for s in stages)
        return AnalysisResult(
            label="live",
            source=f"live://{node.get_name()}",
            duration_s=obs_duration_s,
            message_count=message_count,
            stages=stages,
            links=links,
            bottleneck_link=bottleneck_link,
            likely_node=likely_node,
            likely_node_publishers=likely_node_publishers,
            counters=counters,
            floats={},
            derived=rescue_derived,
            hints=hints,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def print_report(res: AnalysisResult):
    print(f"\n=== {res.label.upper()} ===")
    print(f"source: {res.source}")
    print(f"duration_s: {res.duration_s:.2f}")
    print(f"message_count: {res.message_count}")
    if not res.stages:
        print("Keine bekannten Pipeline-Topics gefunden.")
        return

    print("\nStage rates")
    print("stage         rate_hz   count    dt_p50_ms   dt_p95_ms   jitter_cv   topic")
    for s in res.stages:
        print(
            f"{s.stage:<12} {s.rate_hz:>7.2f} {s.count:>8d} "
            f"{fmt(s.dt_p50_ms, 2):>10} {fmt(s.dt_p95_ms, 2):>10} {fmt(s.jitter_cv, 3):>10}   {s.topic}"
        )
        if s.topic_type:
            print(f"  type: {s.topic_type}")
        if s.publisher_nodes:
            print(f"  publishers: {', '.join(s.publisher_nodes)}")

    print("\nLink throughput ratios (downstream/upstream)")
    if not res.links:
        print("n/a")
    for link in res.links:
        mark = ""
        if res.bottleneck_link is not None and (
            link.upstream_stage == res.bottleneck_link.upstream_stage
            and link.downstream_stage == res.bottleneck_link.downstream_stage
        ):
            mark = "  <-- bottleneck"
        print(
            f"{link.upstream_stage:>10} -> {link.downstream_stage:<10}: {fmt(link.ratio, 3)}{mark}"
        )

    if res.counters or res.floats or res.derived:
        print("\nSecondary signals")
        for k in sorted(res.counters.keys()):
            v = res.counters[k]
            print(f"{k}: {'n/a' if v is None else v}")
        for k in sorted(res.floats.keys()):
            v = res.floats[k]
            print(f"{k}_mean: {fmt(v, 4)}")
        for k in sorted(res.derived.keys()):
            v = res.derived[k]
            print(f"{k}: {fmt(v, 4)}")

    if res.bottleneck_link is not None:
        print(
            "\nLikely bottleneck: "
            f"{res.bottleneck_link.upstream_stage} -> {res.bottleneck_link.downstream_stage} "
            f"(ratio={fmt(res.bottleneck_link.ratio, 3)})"
        )
        if res.likely_node:
            print(f"Likely slow node: {res.likely_node}")
        if res.likely_node_publishers:
            print(f"Observed publisher node(s): {', '.join(res.likely_node_publishers)}")
    else:
        print("\nLikely bottleneck: n/a (nicht genug Daten)")

    if res.hints:
        print("Hints:")
        for h in res.hints:
            print(f"- {h}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Identify likely IBVS pipeline bottleneck from rosbag2 sqlite data "
            "or live ROS topics."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        help=(
            "Path to .db3, bag directory, or benchmark run directory "
            "(with off/shadow/active subfolders). Not needed with --live."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Analyze running ROS system live (topic rates) instead of bag files.",
    )
    parser.add_argument(
        "--disable-stage",
        action="append",
        choices=PIPELINE_STAGE_NAMES,
        default=[],
        help="Disable one pipeline stage (can be repeated), e.g. --disable-stage twist_cmd",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Live observation duration in seconds (default: 15).",
    )
    parser.add_argument(
        "--live-discovery-wait",
        type=float,
        default=3.0,
        help="How long live mode waits for DDS topic discovery before subscribing.",
    )
    parser.add_argument(
        "--live-refresh",
        type=float,
        default=1.0,
        help="Live table refresh interval in seconds (<=0 disables refreshing table).",
    )
    parser.add_argument(
        "--camera-topic",
        default="",
        help="Optional override for camera stage topic in live mode.",
    )
    parser.add_argument(
        "--keypoints-topic",
        default="",
        help="Optional override for keypoints stage topic in live mode.",
    )
    parser.add_argument(
        "--matches-topic",
        default="",
        help="Optional override for matches stage topic in live mode.",
    )
    parser.add_argument(
        "--filtered-topic",
        default="",
        help="Optional override for filtered stage topic in live mode.",
    )
    parser.add_argument(
        "--twist-topic",
        default="",
        help="Optional override for twist_cmd stage topic in live mode.",
    )
    parser.add_argument(
        "--local-rescue-attempts-topic",
        default="",
        help="Optional override for local rescue attempts topic in live mode.",
    )
    parser.add_argument(
        "--local-rescue-success-topic",
        default="",
        help="Optional override for local rescue success topic in live mode.",
    )
    parser.add_argument(
        "--local-rescue-reject-topic",
        default="",
        help="Optional override for local rescue reject topic in live mode.",
    )
    parser.add_argument(
        "--local-rescue-debug-topic",
        default="",
        help="Optional override for local rescue debug topic in live mode.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "off", "shadow", "active"],
        help="Mode selection for benchmark run directories (default: auto).",
    )
    parser.add_argument(
        "--expected-camera-fps",
        type=float,
        default=30.0,
        help="Expected camera fps for low-input warning (default: 30).",
    )
    parser.add_argument(
        "--min-stage-rate-hz",
        type=float,
        default=0.5,
        help="Ignore links with upstream rate below this threshold for bottleneck pick.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output json file.",
    )
    args = parser.parse_args()

    results: List[AnalysisResult] = []
    if args.live:
        try:
            live_topic_overrides = {
                "camera": str(args.camera_topic or "").strip(),
                "keypoints": str(args.keypoints_topic or "").strip(),
                "matches": str(args.matches_topic or "").strip(),
                "filtered": str(args.filtered_topic or "").strip(),
                "twist_cmd": str(args.twist_topic or "").strip(),
            }
            rescue_topic_overrides = {
                "local_rescue_attempts": str(args.local_rescue_attempts_topic or "").strip(),
                "local_rescue_success": str(args.local_rescue_success_topic or "").strip(),
                "local_rescue_reject": str(args.local_rescue_reject_topic or "").strip(),
                "local_rescue_debug": str(args.local_rescue_debug_topic or "").strip(),
            }
            res = analyze_live(
                duration_s=float(args.duration),
                expected_camera_fps=float(args.expected_camera_fps),
                min_stage_rate_hz=float(args.min_stage_rate_hz),
                live_topic_overrides=live_topic_overrides,
                discovery_wait_s=float(args.live_discovery_wait),
                disabled_stages=list(args.disable_stage or []),
                live_refresh_s=float(args.live_refresh),
                rescue_topic_overrides=rescue_topic_overrides,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        print_report(res)
        results.append(res)
    else:
        if not args.path:
            raise SystemExit("path argument required unless --live is set")
        try:
            inputs = resolve_inputs(args.path, args.mode)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        for label, db_path in inputs.items():
            res = analyze_db(
                db_path=db_path,
                label=label,
                expected_camera_fps=float(args.expected_camera_fps),
                min_stage_rate_hz=float(args.min_stage_rate_hz),
                disabled_stages=list(args.disable_stage or []),
            )
            print_report(res)
            results.append(res)

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = [asdict(r) for r in results]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote json: {out_path}")


if __name__ == "__main__":
    main()
