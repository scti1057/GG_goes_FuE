from __future__ import annotations

import threading
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from ibvs_msgs.msg import Keypoints, Matches
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (n + eps)


class LocalRescueDebugNode(Node):
    def __init__(self):
        super().__init__("local_rescue_debug_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("keypoints_topic", "/ibvs/keypoints")
        self.declare_parameter("reference_topic", "/ibvs/reference/keypoints")
        self.declare_parameter("filtered_matches_topic", "/ibvs/filtered_features")
        self.declare_parameter("filter_uncertainty_topic", "/ibvs/filter/uncertainty")
        self.declare_parameter("output_topic", "/ibvs/debug/local_rescue_image")

        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("filtered_timeout_sec", 1.0)

        # Must mirror descriptor_matcher parameters for comparable visualization.
        self.declare_parameter("local_search_radius_px", 18.0)
        self.declare_parameter("local_match_threshold", 0.72)
        self.declare_parameter("sim_floor", 0.60)
        self.declare_parameter("use_adaptive_gates", True)
        self.declare_parameter("adaptive_radius_min_px", 8.0)
        self.declare_parameter("adaptive_radius_max_px", 24.0)
        self.declare_parameter("adaptive_sim_threshold_min", 0.68)
        self.declare_parameter("adaptive_sim_threshold_max", 0.82)
        self.declare_parameter("kp_sigma_low_px", 1.5)
        self.declare_parameter("kp_sigma_high_px", 10.0)
        self.declare_parameter("local_ambiguity_min_score_gap", 0.06)
        self.declare_parameter("local_ambiguity_min_score_ratio", 1.10)

        self.declare_parameter("focus_ref_id", -1)
        self.declare_parameter("max_refs_to_draw", 8)
        self.declare_parameter("max_candidates_per_ref", 8)
        self.declare_parameter("show_candidate_scores", True)
        self.declare_parameter("sync_matcher_params", True)
        self.declare_parameter("matcher_node_name", "/descriptor_matcher_node")
        self.declare_parameter("matcher_param_poll_hz", 2.0)

        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        self.ref_desc: Optional[np.ndarray] = None
        self.ref_d = 0

        self.kpts_xy = np.zeros((0, 2), dtype=np.float32)
        self.kpts_desc = np.zeros((0, 0), dtype=np.float32)

        self.filtered_ref_ids = np.zeros((0,), dtype=np.int64)
        self.filtered_xy = np.zeros((0, 2), dtype=np.float32)
        self.filtered_sigma_px = np.zeros((0,), dtype=np.float32)
        self.filtered_stamp_sec = -1.0

        self.filter_trace = 0.0
        self.last_publish_sec = -1.0
        self.synced_params = {}
        self.matcher_param_client = None
        self.matcher_param_future = None
        self.matcher_sync_warned = False
        self.matcher_synced_param_names = [
            "local_search_radius_px",
            "local_match_threshold",
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

        image_topic = str(self.get_parameter("image_topic").value)
        keypoints_topic = str(self.get_parameter("keypoints_topic").value)
        reference_topic = str(self.get_parameter("reference_topic").value)
        filtered_matches_topic = str(self.get_parameter("filtered_matches_topic").value)
        filter_unc_topic = str(self.get_parameter("filter_uncertainty_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        ref_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Keypoints, reference_topic, self.on_reference, ref_qos)
        self.create_subscription(Keypoints, keypoints_topic, self.on_keypoints, qos_profile_sensor_data)
        self.create_subscription(Matches, filtered_matches_topic, self.on_filtered_matches, qos_profile_sensor_data)
        self.create_subscription(Float32, filter_unc_topic, self.on_filter_uncertainty, qos_profile_sensor_data)

        image_msg_type = CompressedImage if self._topic_uses_compressed(image_topic) else Image
        self.create_subscription(image_msg_type, image_topic, self.on_image, qos_profile_sensor_data)

        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f"Sub image:       {image_topic}")
        self.get_logger().info(f"Sub keypoints:   {keypoints_topic}")
        self.get_logger().info(f"Sub reference:   {reference_topic}")
        self.get_logger().info(f"Sub filtered:    {filtered_matches_topic}")
        self.get_logger().info(f"Sub uncertainty: {filter_unc_topic}")
        self.get_logger().info(f"Pub debug:       {output_topic}")

        if bool(self.get_parameter("sync_matcher_params").value):
            matcher_node_name = str(self.get_parameter("matcher_node_name").value)
            service_name = f"{matcher_node_name.rstrip('/')}/get_parameters"
            self.matcher_param_client = self.create_client(GetParameters, service_name)
            poll_hz = max(0.1, float(self.get_parameter("matcher_param_poll_hz").value))
            self.create_timer(1.0 / poll_hz, self._sync_matcher_params_timer)
            self.get_logger().info(
                f"Parameter sync enabled from {service_name} at {poll_hz:.2f} Hz"
            )

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _p(self, name: str):
        if bool(self.get_parameter("sync_matcher_params").value):
            with self.lock:
                if name in self.synced_params:
                    return self.synced_params[name]
        return self.get_parameter(name).value

    @staticmethod
    def _param_value_to_python(param_value):
        t = int(param_value.type)
        if t == ParameterType.PARAMETER_BOOL:
            return bool(param_value.bool_value)
        if t == ParameterType.PARAMETER_INTEGER:
            return int(param_value.integer_value)
        if t == ParameterType.PARAMETER_DOUBLE:
            return float(param_value.double_value)
        if t == ParameterType.PARAMETER_STRING:
            return str(param_value.string_value)
        if t == ParameterType.PARAMETER_BYTE_ARRAY:
            return list(param_value.byte_array_value)
        if t == ParameterType.PARAMETER_BOOL_ARRAY:
            return list(param_value.bool_array_value)
        if t == ParameterType.PARAMETER_INTEGER_ARRAY:
            return list(param_value.integer_array_value)
        if t == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return list(param_value.double_array_value)
        if t == ParameterType.PARAMETER_STRING_ARRAY:
            return list(param_value.string_array_value)
        return None

    def _sync_matcher_params_timer(self):
        if not bool(self.get_parameter("sync_matcher_params").value):
            return
        if self.matcher_param_client is None:
            return

        if self.matcher_param_future is not None:
            if not self.matcher_param_future.done():
                return
            self.matcher_param_future = None

        if not self.matcher_param_client.service_is_ready():
            if not self.matcher_sync_warned:
                self.get_logger().warn(
                    "Matcher parameter service not ready yet; using local debug params."
                )
                self.matcher_sync_warned = True
            return

        self.matcher_sync_warned = False
        req = GetParameters.Request()
        req.names = self.matcher_synced_param_names
        self.matcher_param_future = self.matcher_param_client.call_async(req)
        self.matcher_param_future.add_done_callback(self._on_matcher_params_synced)

    def _on_matcher_params_synced(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Matcher parameter sync failed: {exc}")
            return

        updates = {}
        for name, value_msg in zip(self.matcher_synced_param_names, resp.values):
            py_val = self._param_value_to_python(value_msg)
            if py_val is not None:
                updates[name] = py_val

        if not updates:
            return
        with self.lock:
            self.synced_params.update(updates)

    @staticmethod
    def _topic_uses_compressed(topic: str) -> bool:
        return topic.endswith("/compressed") or topic.endswith("/compressedDepth")

    def _decode_color_bgr(self, msg: Any) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            try:
                bgr = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                bgr = None
            if bgr is None:
                bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("compressed image decode returned None")
            return bgr
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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

    def _consume_publish_slot(self) -> bool:
        now_sec = self._now_sec()
        min_period = 1.0 / max(float(self._p("publish_rate_hz")), 1e-3)
        with self.lock:
            if self.last_publish_sec > 0.0 and (now_sec - self.last_publish_sec) < min_period:
                return False
            self.last_publish_sec = now_sec
        return True

    def _normalize_uncertainty(self, sigma_px: float) -> float:
        low = float(self._p("kp_sigma_low_px"))
        high = float(self._p("kp_sigma_high_px"))
        if high <= low + 1e-9:
            return 0.5
        u = (float(sigma_px) - low) / (high - low)
        # print(f"Normalized uncertainty u={u:.2f} from sigma_px={sigma_px:.2f} with low={low:.2f} and high={high:.2f}")
        return float(np.clip(u, 0.0, 1.0))

    def _effective_gates(self, u: float):
        if bool(self._p("use_adaptive_gates")):
            r_min = float(self._p("adaptive_radius_min_px"))
            r_max = float(self._p("adaptive_radius_max_px"))
            t_min = float(self._p("adaptive_sim_threshold_min"))
            t_max = float(self._p("adaptive_sim_threshold_max"))
            radius = r_min + u * (r_max - r_min)
            sim_thr = t_min + u * (t_max - t_min)
        else:
            radius = float(self._p("local_search_radius_px"))
            sim_thr = float(self._p("local_match_threshold"))

        sim_floor = float(self._p("sim_floor"))
        radius = max(0.1, float(radius))
        sim_thr = max(sim_floor, min(1.0, float(sim_thr)))
        return radius, sim_thr

    def on_reference(self, msg: Keypoints):
        d = int(msg.descriptor_dim)
        xy = np.asarray(msg.xy, dtype=np.float32)
        if d <= 0 or xy.size % 2 != 0:
            return

        k = xy.reshape(-1, 2)
        if len(msg.descriptors) != k.shape[0] * d:
            return

        desc = np.asarray(msg.descriptors, dtype=np.float32).reshape(k.shape[0], d)
        with self.lock:
            self.ref_desc = l2_normalize(desc.astype(np.float32))
            self.ref_d = d

    def on_keypoints(self, msg: Keypoints):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            return
        cur_xy = xy.reshape(-1, 2)
        n = cur_xy.shape[0]

        d = int(msg.descriptor_dim)
        if d <= 0 or len(msg.descriptors) != n * d:
            with self.lock:
                self.kpts_xy = cur_xy
                self.kpts_desc = np.zeros((0, 0), dtype=np.float32)
            return

        desc = np.asarray(msg.descriptors, dtype=np.float32).reshape(n, d)
        desc = l2_normalize(desc.astype(np.float32))

        with self.lock:
            self.kpts_xy = cur_xy
            self.kpts_desc = desc

    def on_filtered_matches(self, msg: Matches):
        ref_ids, cur_xy, sigma = self._parse_matches(msg)
        with self.lock:
            self.filtered_ref_ids = ref_ids
            self.filtered_xy = cur_xy
            self.filtered_sigma_px = sigma
            self.filtered_stamp_sec = self._now_sec()

    def on_filter_uncertainty(self, msg: Float32):
        with self.lock:
            self.filter_trace = float(msg.data)

    def on_image(self, msg: Any):
        if not self._consume_publish_slot():
            return

        try:
            cv_img = self._decode_color_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"image decode failed: {exc}")
            return

        with self.lock:
            ref_desc = None if self.ref_desc is None else self.ref_desc.copy()
            ref_d = int(self.ref_d)
            kpts_xy = self.kpts_xy.copy()
            kpts_desc = self.kpts_desc.copy()
            filtered_ref_ids = self.filtered_ref_ids.copy()
            filtered_xy = self.filtered_xy.copy()
            filtered_sigma = self.filtered_sigma_px.copy()
            filtered_stamp_sec = float(self.filtered_stamp_sec)
            filter_trace = float(self.filter_trace)

        self._draw_overlay(
            cv_img,
            ref_desc,
            ref_d,
            kpts_xy,
            kpts_desc,
            filtered_ref_ids,
            filtered_xy,
            filtered_sigma,
            filtered_stamp_sec,
            filter_trace,
        )

        out = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)

    def _draw_overlay(
        self,
        cv_img: np.ndarray,
        ref_desc: Optional[np.ndarray],
        ref_d: int,
        kpts_xy: np.ndarray,
        kpts_desc: np.ndarray,
        filtered_ref_ids: np.ndarray,
        filtered_xy: np.ndarray,
        filtered_sigma: np.ndarray,
        filtered_stamp_sec: float,
        filter_trace: float,
    ):
        h, w = cv_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        sim_floor = float(self._p("sim_floor"))
        focus_ref_id = int(self._p("focus_ref_id"))
        max_refs = max(1, int(self._p("max_refs_to_draw")))
        max_candidates = max(1, int(self._p("max_candidates_per_ref")))
        show_scores = bool(self._p("show_candidate_scores"))
        timeout_sec = max(1e-3, float(self._p("filtered_timeout_sec")))
        min_gap = max(0.0, float(self._p("local_ambiguity_min_score_gap")))
        min_ratio = max(1.0, float(self._p("local_ambiguity_min_score_ratio")))

        stale = True
        if filtered_stamp_sec > 0.0:
            stale = (self._now_sec() - filtered_stamp_sec) > timeout_sec

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (8, 8), (560, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, cv_img, 0.55, 0, cv_img)

        y = 28
        cv2.putText(cv_img, "Local Rescue Debug", (16, y), font, 0.55, (255, 255, 255), 1)
        y += 20
        cv2.putText(
            cv_img,
            f"sim_floor={sim_floor:.2f} trace(P)={filter_trace:.2f} stale={stale}",
            (16, y),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            cv_img,
            f"kpts={kpts_xy.shape[0]} filtered_active={filtered_ref_ids.size} focus_ref_id={focus_ref_id}",
            (16, y),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            cv_img,
            f"ambiguity: gap>={min_gap:.2f}, ratio>={min_ratio:.2f}",
            (16, y),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        if stale or filtered_ref_ids.size <= 0:
            cv2.putText(cv_img, "No fresh filtered predictions available.", (16, 175), font, 0.6, (0, 200, 255), 2)
            return

        if (
            ref_desc is None
            or ref_desc.shape[0] <= 0
            or ref_d <= 0
            or kpts_xy.shape[0] <= 0
            or kpts_desc.shape[0] != kpts_xy.shape[0]
        ):
            cv2.putText(cv_img, "Missing reference/keypoint descriptors.", (16, 175), font, 0.6, (0, 200, 255), 2)
            return

        if focus_ref_id >= 0:
            draw_indices = [i for i in range(filtered_ref_ids.size) if int(filtered_ref_ids[i]) == focus_ref_id]
        else:
            draw_indices = list(range(min(filtered_ref_ids.size, max_refs)))

        if not draw_indices:
            cv2.putText(cv_img, "focus_ref_id not active in filtered set.", (16, 175), font, 0.55, (0, 200, 255), 2)
            return

        n_f = min(filtered_ref_ids.size, filtered_xy.shape[0], filtered_sigma.size)
        draw_indices = [i for i in draw_indices if i < n_f]

        for idx in draw_indices:
            rid = int(filtered_ref_ids[idx])
            if rid < 0 or rid >= ref_desc.shape[0]:
                continue

            pred = filtered_xy[idx]
            sigma_px = float(filtered_sigma[idx])
            u = self._normalize_uncertainty(sigma_px)
            radius_eff, sim_thr_eff = self._effective_gates(u)
            radius2 = radius_eff * radius_eff

            px, py = int(round(pred[0])), int(round(pred[1]))
            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            cv2.circle(cv_img, (px, py), int(round(radius_eff)), (0, 255, 255), 1)
            cv2.drawMarker(cv_img, (px, py), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            d2 = np.sum((kpts_xy - pred[None, :]) ** 2, axis=1)
            cand = np.where(d2 <= radius2)[0]
            if cand.size <= 0:
                cv2.putText(cv_img, f"rid={rid}: no candidates", (px + 6, py - 8), font, 0.38, (0, 200, 255), 1)
                continue

            sims = kpts_desc[cand] @ ref_desc[rid]
            dists = np.sqrt(d2[cand])
            valid = (sims >= sim_floor) & (sims >= sim_thr_eff)
            if not np.any(valid):
                cv2.putText(cv_img, f"rid={rid}: all fail gates", (px + 6, py - 8), font, 0.38, (0, 120, 255), 1)
                continue

            cand_v = cand[valid]
            sims_v = sims[valid]
            dists_v = dists[valid]

            sim_norm = np.clip((sims_v - sim_floor) / max(1e-6, (1.0 - sim_floor)), 0.0, 1.0)
            sigma_pos = max(1.0, 0.5 * radius_eff)
            pos_score = np.exp(-0.5 * (dists_v / sigma_pos) ** 2)
            w_pos = 1.0 - u
            w_sim = u
            score = np.power(pos_score, w_pos) * np.power(sim_norm, w_sim)

            order = np.argsort(-score)
            order = order[:max_candidates]

            best = int(order[0])
            best_idx = int(cand_v[best])
            best_sim = float(sims_v[best])
            best_score = float(score[best])
            best_pt = kpts_xy[best_idx]

            ambiguous = False
            if score.size > 1:
                second = int(np.argsort(-score)[1])
                second_score = float(score[second])
                gap = best_score - second_score
                ratio = best_score / max(second_score, 1e-6)
                ambiguous = (gap < min_gap) or (ratio < min_ratio)

            for rank, pos in enumerate(order.tolist()):
                cidx = int(cand_v[int(pos)])
                sim = float(sims_v[int(pos)])
                sc = float(score[int(pos)])
                pt = kpts_xy[cidx]
                cx, cy = int(round(pt[0])), int(round(pt[1]))

                color = (0, 220, 0) if sim >= sim_thr_eff else (0, 0, 220)
                cv2.circle(cv_img, (cx, cy), 3, color, -1)
                if show_scores:
                    cv2.putText(cv_img, f"s={sim:.2f} q={sc:.2f}", (cx + 3, cy - 3 - 10 * min(rank, 2)), font, 0.32, color, 1)

            bx, by = int(round(best_pt[0])), int(round(best_pt[1]))
            if ambiguous:
                pass_color = (0, 140, 255)
                status = "AMB"
            else:
                pass_color = (0, 255, 0)
                status = "OK"

            cv2.line(cv_img, (px, py), (bx, by), pass_color, 1)
            cv2.circle(cv_img, (bx, by), 6, pass_color, 1)
            cv2.putText(
                cv_img,
                f"rid={rid} sig={sigma_px:.2f} u={u:.2f} r={radius_eff:.1f} thr={sim_thr_eff:.2f} best={best_sim:.2f}/{best_score:.2f} {status}",
                (px + 6, py + 12),
                font,
                0.37,
                pass_color,
                1,
            )
        self._draw_coordinate_axes(cv_img)
            
    @staticmethod
    def _draw_coordinate_axes(cv_img):
        h, w = cv_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Compact panel in the upper-right corner.
        margin = 10
        panel_w, panel_h = 110, 85
        panel_tl = (max(0, w - panel_w - margin), margin)
        panel_br = (min(w - 1, panel_tl[0] + panel_w), min(h - 1, panel_tl[1] + panel_h))

        # Origin is near the lower-right inside this panel so +x left and +y up are visible.
        origin = (panel_br[0]-40, panel_br[1] - 18)
        axis_len = 30

        # Background panel for readability.
        overlay = cv_img.copy()
        cv2.rectangle(overlay, panel_tl, panel_br, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, cv_img, 0.65, 0, cv_img)

        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)

        # +x points to the left in the image.
        cv2.arrowedLine(
            cv_img,
            origin,
            (origin[0] - axis_len, origin[1]),
            color_x,
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            cv_img,
            '+x',
            (origin[0] - axis_len - 20, origin[1] - 2),
            font,
            0.38,
            color_x,
            1,
            cv2.LINE_AA,
        )

        # +y points upwards in the image.
        cv2.arrowedLine(
            cv_img,
            origin,
            (origin[0], origin[1] - axis_len),
            color_y,
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            cv_img,
            '+y',
            (origin[0] + 4, origin[1] - axis_len - 4),
            font,
            0.38,
            color_y,
            1,
            cv2.LINE_AA,
        )

        # +z points into the image plane (circle with cross) at the axis origin.
        z_center = origin
        z_r = 5
        cv2.circle(cv_img, z_center, z_r, color_z, 1, cv2.LINE_AA)
        cv2.line(
            cv_img,
            (z_center[0] - 3, z_center[1] - 3),
            (z_center[0] + 3, z_center[1] + 3),
            color_z,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            cv_img,
            (z_center[0] - 3, z_center[1] + 3),
            (z_center[0] + 3, z_center[1] - 3),
            color_z,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            cv_img,
            '+z',
            (z_center[0] + 8, z_center[1] + 12),
            font,
            0.38,
            color_z,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            cv_img,
            'Koordsystem',
            (panel_tl[0] + 6, panel_tl[1] + 14),
            font,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def main():
    rclpy.init()
    node = LocalRescueDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
