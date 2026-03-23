import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String, UInt32

from ibvs_msgs.msg import (
    Keypoints,
    LocalRescueDebug,
    LocalRescueDebugCandidate,
    LocalRescueDebugEntry,
    Matches,
)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (n + eps)


class DescriptorMatcherNode(Node):
    def __init__(self):
        super().__init__("descriptor_matcher_node")

        self.declare_parameter("keypoints_topic", "/ibvs/keypoints")
        self.declare_parameter("reference_topic", "/ibvs/reference/keypoints")
        self.declare_parameter("matches_topic", "/ibvs/matches")
        self.declare_parameter("filtered_matches_topic", "/ibvs/filtered_features")

        self.declare_parameter("match_threshold", 0.85)
        self.declare_parameter("mutual_check", True)

        # Runtime switch for benchmarking.
        self.declare_parameter("local_rescue_mode", "off")  # off | shadow | active
        self.declare_parameter("max_local_rescues_per_frame", 12)
        self.declare_parameter("filtered_timeout_sec", 0.25)

        # Legacy fixed gates (used when adaptive gates are disabled).
        self.declare_parameter("local_search_radius_px", 18.0)
        self.declare_parameter("local_match_threshold", 0.72)

        # Hard and adaptive gates.
        self.declare_parameter("sim_floor", 0.60)
        self.declare_parameter("use_adaptive_gates", True)
        self.declare_parameter("adaptive_radius_min_px", 8.0)
        self.declare_parameter("adaptive_radius_max_px", 40.0)
        self.declare_parameter("adaptive_sim_threshold_min", 0.65)
        self.declare_parameter("adaptive_sim_threshold_max", 0.95)

        # Per-keypoint uncertainty normalization (sigma_px from filtered_features.sim).
        self.declare_parameter("kp_sigma_low_px", 1.5)
        self.declare_parameter("kp_sigma_high_px", 25.0)

        # Ambiguity checks on the combined score.
        self.declare_parameter("local_ambiguity_min_score_gap", 0.06)
        self.declare_parameter("local_ambiguity_min_score_ratio", 1.10)

        # Optional global uncertainty gate.
        self.declare_parameter("use_uncertainty_gate", False)
        self.declare_parameter("filter_uncertainty_topic", "/ibvs/filter/uncertainty")
        self.declare_parameter("max_filter_trace_for_rescue", 1.0e9)

        self.declare_parameter("publish_local_rescue_stats", True)
        self.declare_parameter("local_rescue_stats_topic", "/ibvs/matching/local_rescue_stats")
        self.declare_parameter("local_rescue_attempts_topic", "/ibvs/matching/local_rescue_attempts")
        self.declare_parameter("local_rescue_success_topic", "/ibvs/matching/local_rescue_success")
        self.declare_parameter("local_rescue_reject_topic", "/ibvs/matching/local_rescue_reject")

        self.declare_parameter("publish_local_rescue_debug", True)
        self.declare_parameter("local_rescue_debug_topic", "/ibvs/matching/local_rescue_debug")
        self.declare_parameter("local_rescue_debug_max_candidates_per_entry", 8)

        self.ref_desc = None
        self.ref_d = 0

        self.latest_filtered_ref_ids = np.zeros((0,), dtype=np.int64)
        self.latest_filtered_xy = np.zeros((0, 2), dtype=np.float32)
        self.latest_filtered_sigma_px = np.zeros((0,), dtype=np.float32)
        self.latest_filtered_time_sec = -1.0
        self.latest_filter_trace = 0.0

        self.total_rescue_attempts = 0
        self.total_rescue_successes = 0
        self.total_rescue_rejects = 0

        kp_topic = str(self.get_parameter("keypoints_topic").value)
        ref_topic = str(self.get_parameter("reference_topic").value)
        out_topic = str(self.get_parameter("matches_topic").value)
        filtered_topic = str(self.get_parameter("filtered_matches_topic").value)
        filter_unc_topic = str(self.get_parameter("filter_uncertainty_topic").value)

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        ref_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_ref = self.create_subscription(Keypoints, ref_topic, self.on_reference, ref_qos)
        self.sub_kp = self.create_subscription(
            Keypoints,
            kp_topic,
            self.on_keypoints,
            qos_profile_sensor_data,
        )
        self.sub_filtered = self.create_subscription(
            Matches,
            filtered_topic,
            self.on_filtered_matches,
            qos_profile_sensor_data,
        )
        self.sub_filter_unc = self.create_subscription(
            Float32,
            filter_unc_topic,
            self.on_filter_uncertainty,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(Matches, out_topic, 10)

        self.pub_stats = None
        self.pub_attempts = None
        self.pub_successes = None
        self.pub_rejects = None
        if bool(self.get_parameter("publish_local_rescue_stats").value):
            self.pub_stats = self.create_publisher(
                String,
                str(self.get_parameter("local_rescue_stats_topic").value),
                10,
            )
            self.pub_attempts = self.create_publisher(
                UInt32,
                str(self.get_parameter("local_rescue_attempts_topic").value),
                10,
            )
            self.pub_successes = self.create_publisher(
                UInt32,
                str(self.get_parameter("local_rescue_success_topic").value),
                10,
            )
            self.pub_rejects = self.create_publisher(
                UInt32,
                str(self.get_parameter("local_rescue_reject_topic").value),
                10,
            )

        self.pub_local_debug = None
        if bool(self.get_parameter("publish_local_rescue_debug").value):
            self.pub_local_debug = self.create_publisher(
                LocalRescueDebug,
                str(self.get_parameter("local_rescue_debug_topic").value),
                10,
            )

        self.get_logger().info(f"Sub keypoints:  {kp_topic}")
        self.get_logger().info(f"Sub reference:  {ref_topic}")
        self.get_logger().info(f"Sub filtered:   {filtered_topic}")
        self.get_logger().info(f"Sub filter unc: {filter_unc_topic}")
        self.get_logger().info(f"Pub matches:    {out_topic}")
        if self.pub_local_debug is not None:
            self.get_logger().info(
                f"Pub rescue dbg: {str(self.get_parameter('local_rescue_debug_topic').value)}"
            )
        self.get_logger().info("Local rescue mode: off|shadow|active")

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    @staticmethod
    def _parse_mode(raw: str) -> str:
        mode = str(raw).strip().lower()
        if mode in ("off", "shadow", "active"):
            return mode
        return "off"

    @staticmethod
    def _parse_matches(msg: Matches):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        cur = xy.reshape(-1, 2)
        n = min(cur.shape[0], len(msg.ref_id))
        if n <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        sigma = np.zeros((n,), dtype=np.float32)
        if len(msg.sim) >= n:
            sigma = np.asarray(msg.sim[:n], dtype=np.float32)
        return np.asarray(msg.ref_id[:n], dtype=np.int64), cur[:n], sigma

    def _normalize_uncertainty(self, sigma_px: float) -> float:
        low = float(self.get_parameter("kp_sigma_low_px").value)
        high = float(self.get_parameter("kp_sigma_high_px").value)
        if not np.isfinite(sigma_px):
            sigma_px = high
        if high <= low + 1e-9:
            return 0.5
        u = (float(sigma_px) - low) / (high - low)
        return float(np.clip(u, 0.0, 1.0))

    def _effective_gates(self, u: float):
        if bool(self.get_parameter("use_adaptive_gates").value):
            r_min = float(self.get_parameter("adaptive_radius_min_px").value)
            r_max = float(self.get_parameter("adaptive_radius_max_px").value)
            t_min = float(self.get_parameter("adaptive_sim_threshold_min").value)
            t_max = float(self.get_parameter("adaptive_sim_threshold_max").value)
            radius = r_min + u * (r_max - r_min)
            sim_thr = t_min + u * (t_max - t_min)
        else:
            radius = float(self.get_parameter("local_search_radius_px").value)
            sim_thr = float(self.get_parameter("local_match_threshold").value)

        sim_floor = float(self.get_parameter("sim_floor").value)
        radius = max(0.1, float(radius))
        sim_thr = max(sim_floor, min(1.0, float(sim_thr)))
        return radius, sim_thr

    def on_reference(self, msg: Keypoints):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn("Reference xy length not even; ignoring.")
            return
        k = xy.reshape(-1, 2)
        d = int(msg.descriptor_dim)

        if d <= 0:
            self.get_logger().warn("Reference has no descriptors (descriptor_dim=0); ignoring.")
            return
        if len(msg.descriptors) != k.shape[0] * d:
            self.get_logger().warn("Reference descriptors size mismatch; ignoring.")
            return

        desc = np.asarray(msg.descriptors, dtype=np.float32).reshape(k.shape[0], d)
        self.ref_desc = l2_normalize(desc.astype(np.float32))
        self.ref_d = d
        self.get_logger().info(f"Reference cached: K={k.shape[0]} D={d}")

    def on_filtered_matches(self, msg: Matches):
        ref_ids, cur_xy, sigma = self._parse_matches(msg)
        self.latest_filtered_ref_ids = ref_ids
        self.latest_filtered_xy = cur_xy
        self.latest_filtered_sigma_px = sigma
        self.latest_filtered_time_sec = self._now_sec()

    def on_filter_uncertainty(self, msg: Float32):
        self.latest_filter_trace = float(msg.data)

    def _local_context_status(self):
        mode = self._parse_mode(self.get_parameter("local_rescue_mode").value)
        timeout = max(1e-3, float(self.get_parameter("filtered_timeout_sec").value))

        filtered_stale = True
        if self.latest_filtered_time_sec > 0.0:
            filtered_stale = (self._now_sec() - self.latest_filtered_time_sec) > timeout

        if mode == "off":
            return False, filtered_stale

        if self.latest_filtered_time_sec <= 0.0:
            return False, True

        if filtered_stale:
            return False, True

        if bool(self.get_parameter("use_uncertainty_gate").value):
            max_trace = float(self.get_parameter("max_filter_trace_for_rescue").value)
            if self.latest_filter_trace > max_trace:
                return False, False

        return True, False

    @staticmethod
    def _build_debug_entry(
        rid: int,
        pred_xy: np.ndarray,
        sigma_px: float,
        u: float,
        radius_eff: float,
        sim_thr_eff: float,
        status: str,
    ) -> LocalRescueDebugEntry:
        entry = LocalRescueDebugEntry()
        entry.ref_id = int(rid) if rid >= 0 else 0
        entry.pred_x = float(pred_xy[0])
        entry.pred_y = float(pred_xy[1])
        entry.sigma_px = float(sigma_px)
        entry.uncertainty_u = float(u)
        entry.radius_eff_px = float(radius_eff)
        entry.sim_threshold_eff = float(sim_thr_eff)
        entry.status = status
        return entry

    def _run_local_rescue(
        self,
        kpts: np.ndarray,
        desc: np.ndarray,
        assigned_cur: set[int],
        assigned_ref: set[int],
    ):
        rescues = []
        attempts = 0
        successes = 0
        debug_entries = []

        if self.ref_desc is None or self.ref_desc.shape[0] <= 0:
            return rescues, attempts, successes, debug_entries
        if self.latest_filtered_ref_ids.size <= 0 or self.latest_filtered_xy.shape[0] <= 0:
            return rescues, attempts, successes, debug_entries

        max_rescues = max(0, int(self.get_parameter("max_local_rescues_per_frame").value))
        if max_rescues <= 0:
            return rescues, attempts, successes, debug_entries

        max_dbg_candidates = max(
            1,
            int(self.get_parameter("local_rescue_debug_max_candidates_per_entry").value),
        )

        sim_floor = float(self.get_parameter("sim_floor").value)
        min_gap = max(0.0, float(self.get_parameter("local_ambiguity_min_score_gap").value))
        min_ratio = max(1.0, float(self.get_parameter("local_ambiguity_min_score_ratio").value))

        n_f = min(
            self.latest_filtered_ref_ids.size,
            self.latest_filtered_xy.shape[0],
            self.latest_filtered_sigma_px.size,
        )
        for i in range(n_f):
            rid = int(self.latest_filtered_ref_ids[i])
            pred_xy = self.latest_filtered_xy[i]
            sigma_px = float(self.latest_filtered_sigma_px[i])
            u = self._normalize_uncertainty(sigma_px)
            radius_eff, sim_thr_eff = self._effective_gates(u)
            entry = self._build_debug_entry(
                rid,
                pred_xy,
                sigma_px,
                u,
                radius_eff,
                sim_thr_eff,
                "SKIPPED",
            )

            if rid in assigned_ref:
                entry.status = "SKIP_ASSIGNED_REF"
                debug_entries.append(entry)
                continue

            if rid < 0 or rid >= self.ref_desc.shape[0]:
                entry.status = "SKIP_INVALID_REF"
                debug_entries.append(entry)
                continue

            radius2 = radius_eff * radius_eff
            d2 = np.sum((kpts - pred_xy[None, :]) ** 2, axis=1)
            cand_all = np.where(d2 <= radius2)[0]
            if cand_all.size <= 0:
                entry.status = "NO_CANDIDATES"
                debug_entries.append(entry)
                continue

            cand_idx = [int(idx) for idx in cand_all.tolist() if int(idx) not in assigned_cur]
            if not cand_idx:
                entry.status = "ALL_ASSIGNED_CUR"
                debug_entries.append(entry)
                continue

            attempts += 1

            cand_arr = np.asarray(cand_idx, dtype=np.int64)
            cand_desc = desc[cand_arr]
            sims = cand_desc @ self.ref_desc[rid]
            dists = np.sqrt(d2[cand_arr])

            valid = (sims >= sim_floor) & (sims >= sim_thr_eff)
            if not np.any(valid):
                entry.status = "ALL_FAIL_GATES"
                debug_entries.append(entry)
                continue

            sims_v = sims[valid]
            dists_v = dists[valid]
            cand_v = cand_arr[valid]

            sim_norm = np.clip((sims_v - sim_floor) / max(1e-6, (1.0 - sim_floor)), 0.0, 1.0)
            sigma_pos = max(1.0, 0.5 * radius_eff)
            pos_score = np.exp(-0.5 * (dists_v / sigma_pos) ** 2)

            w_pos = 1.0 - u
            w_sim = u
            score = np.power(pos_score, w_pos) * np.power(sim_norm, w_sim)

            order = np.argsort(-score)
            dbg_order = order[:max_dbg_candidates]
            for pos in dbg_order.tolist():
                cidx = int(cand_v[int(pos)])
                cand = LocalRescueDebugCandidate()
                cand.x = float(kpts[cidx, 0])
                cand.y = float(kpts[cidx, 1])
                cand.sim = float(sims_v[int(pos)])
                cand.score = float(score[int(pos)])
                entry.candidates.append(cand)

            best = int(order[0])
            best_idx = int(cand_v[best])
            best_sim = float(sims_v[best])
            best_score = float(score[best])

            entry.best_x = float(kpts[best_idx, 0])
            entry.best_y = float(kpts[best_idx, 1])
            entry.best_sim = best_sim
            entry.best_score = best_score

            if score.size > 1:
                second = int(order[1])
                second_score = float(score[second])
                gap = best_score - second_score
                ratio = best_score / max(second_score, 1e-6)
                entry.second_score = second_score
                entry.score_gap = gap
                entry.score_ratio = ratio
                if (gap < min_gap) or (ratio < min_ratio):
                    entry.status = "AMBIGUOUS"
                    debug_entries.append(entry)
                    continue

            assigned_cur.add(best_idx)
            assigned_ref.add(rid)
            rescues.append((rid, best_idx, best_sim))
            successes += 1

            entry.status = "RESCUED"
            debug_entries.append(entry)

            if len(rescues) >= max_rescues:
                break

        return rescues, attempts, successes, debug_entries

    def _publish_local_rescue_debug(
        self,
        header,
        mode: str,
        context_ready: bool,
        filtered_stale: bool,
        global_count: int,
        final_count: int,
        local_attempts: int,
        local_successes: int,
        local_applied: int,
        entries,
    ):
        if self.pub_local_debug is None:
            return

        dbg = LocalRescueDebug()
        dbg.header = header
        dbg.mode = mode
        dbg.context_ready = bool(context_ready)
        dbg.filtered_stale = bool(filtered_stale)
        dbg.filter_trace = float(self.latest_filter_trace)
        dbg.sim_floor = float(self.get_parameter("sim_floor").value)
        dbg.ambiguity_min_gap = float(self.get_parameter("local_ambiguity_min_score_gap").value)
        dbg.ambiguity_min_ratio = float(self.get_parameter("local_ambiguity_min_score_ratio").value)
        dbg.global_count = int(max(0, global_count))
        dbg.final_count = int(max(0, final_count))
        dbg.local_attempts = int(max(0, local_attempts))
        dbg.local_successes = int(max(0, local_successes))
        dbg.local_applied = int(max(0, local_applied))
        dbg.entries = list(entries)
        self.pub_local_debug.publish(dbg)

    def _publish_stats(
        self,
        mode: str,
        global_count: int,
        final_count: int,
        local_attempts: int,
        local_successes: int,
        local_applied: int,
    ):
        if self.pub_stats is None:
            return

        frame_rejects = max(0, local_attempts - local_successes)
        self.total_rescue_attempts += int(max(0, local_attempts))
        self.total_rescue_successes += int(max(0, local_successes))
        self.total_rescue_rejects += int(frame_rejects)

        msg = String()
        msg.data = (
            f"mode={mode} global={global_count} final={final_count} "
            f"local_attempts={local_attempts} local_successes={local_successes} "
            f"local_applied={local_applied}"
        )
        self.pub_stats.publish(msg)

        m_attempts = UInt32()
        m_attempts.data = int(self.total_rescue_attempts)
        self.pub_attempts.publish(m_attempts)

        m_success = UInt32()
        m_success.data = int(self.total_rescue_successes)
        self.pub_successes.publish(m_success)

        m_reject = UInt32()
        m_reject.data = int(self.total_rescue_rejects)
        self.pub_rejects.publish(m_reject)

    def on_keypoints(self, msg: Keypoints):
        if self.ref_desc is None:
            return

        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn("Keypoints xy length not even; skipping frame.")
            return
        kpts = xy.reshape(-1, 2)
        n = kpts.shape[0]
        if n == 0:
            return

        if len(msg.depth_m) == n:
            kpts_depth_m = np.asarray(msg.depth_m, dtype=np.float32)
        else:
            kpts_depth_m = np.full((n,), np.nan, dtype=np.float32)

        d = int(msg.descriptor_dim)
        if d != self.ref_d or d <= 0:
            self.get_logger().warn(f"Descriptor dim mismatch: got {d}, ref {self.ref_d}.")
            return
        if len(msg.descriptors) != n * d:
            self.get_logger().warn("Keypoints descriptors size mismatch; skipping frame.")
            return

        desc = np.asarray(msg.descriptors, dtype=np.float32).reshape(n, d)
        desc = l2_normalize(desc.astype(np.float32))

        # Global matching stage (unchanged baseline behavior).
        sim_mat = desc @ self.ref_desc.T
        best_ref = np.argmax(sim_mat, axis=1)
        best_sim = sim_mat[np.arange(n), best_ref]

        thr = float(self.get_parameter("match_threshold").value)
        mutual = bool(self.get_parameter("mutual_check").value)

        keep = best_sim >= thr
        idx = np.where(keep)[0]
        if idx.size > 0 and mutual:
            best_cur_for_ref = np.argmax(sim_mat, axis=0)
            mutual_mask = np.array([best_cur_for_ref[best_ref[i]] == i for i in idx], dtype=bool)
            idx = idx[mutual_mask]

        out_ref = []
        out_xy = []
        out_depth_m = []
        out_sim = []
        assigned_cur: set[int] = set()
        assigned_ref: set[int] = set()

        for cur_idx in idx.tolist():
            rid = int(best_ref[cur_idx])
            out_ref.append(rid)
            out_xy.append(kpts[cur_idx])
            out_depth_m.append(float(kpts_depth_m[cur_idx]))
            out_sim.append(float(best_sim[cur_idx]))
            assigned_cur.add(int(cur_idx))
            assigned_ref.add(rid)

        global_count = len(out_ref)

        mode = self._parse_mode(self.get_parameter("local_rescue_mode").value)
        local_attempts = 0
        local_successes = 0
        local_applied = 0
        debug_entries = []

        local_context_ready, filtered_stale = self._local_context_status()
        if local_context_ready:
            rescues, local_attempts, local_successes, debug_entries = self._run_local_rescue(
                kpts,
                desc,
                assigned_cur,
                assigned_ref,
            )
            if mode == "active":
                for rid, cur_idx, sim in rescues:
                    out_ref.append(int(rid))
                    out_xy.append(kpts[int(cur_idx)])
                    out_depth_m.append(float(kpts_depth_m[int(cur_idx)]))
                    out_sim.append(float(sim))
                local_applied = len(rescues)

        final_count = len(out_ref)
        self._publish_local_rescue_debug(
            msg.header,
            mode,
            local_context_ready,
            filtered_stale,
            global_count,
            final_count,
            local_attempts,
            local_successes,
            local_applied,
            debug_entries,
        )

        if not out_ref:
            self._publish_stats(
                mode,
                global_count,
                0,
                local_attempts,
                local_successes,
                local_applied,
            )
            return

        out = Matches()
        out.header = msg.header
        out.ref_id = np.asarray(out_ref, dtype=np.uint32).tolist()
        out.xy = np.asarray(out_xy, dtype=np.float32).reshape(-1).tolist()
        out.depth_m = np.asarray(out_depth_m, dtype=np.float32).tolist()
        out.sim = np.asarray(out_sim, dtype=np.float32).tolist()
        self.pub.publish(out)

        self._publish_stats(
            mode,
            global_count,
            len(out_ref),
            local_attempts,
            local_successes,
            local_applied,
        )


def main():
    rclpy.init()
    node = DescriptorMatcherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
