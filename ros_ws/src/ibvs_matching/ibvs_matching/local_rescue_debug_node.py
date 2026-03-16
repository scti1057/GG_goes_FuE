from __future__ import annotations

import threading
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from ibvs_msgs.msg import LocalRescueDebug
from rcl_interfaces.msg import ParameterEvent, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class LocalRescueDebugNode(Node):
    def __init__(self):
        super().__init__("local_rescue_debug_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("local_rescue_debug_topic", "/ibvs/matching/local_rescue_debug")
        self.declare_parameter("output_topic", "/ibvs/debug/local_rescue_image")

        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("debug_timeout_sec", 1.0)

        self.declare_parameter("focus_ref_id", -1)
        self.declare_parameter("max_refs_to_draw", 10)
        self.declare_parameter("max_candidates_per_ref", 8)
        self.declare_parameter("show_candidate_scores", True)

        self.declare_parameter("matcher_node_name", "/descriptor_matcher_node")
        self.declare_parameter("show_parameter_events", True)
        self.declare_parameter("param_event_memory_sec", 8.0)

        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        self.last_publish_sec = -1.0
        self.last_debug_msg: Optional[LocalRescueDebug] = None
        self.last_debug_recv_sec = -1.0
        self.recent_param_events: list[tuple[float, str]] = []

        image_topic = str(self.get_parameter("image_topic").value)
        debug_topic = str(self.get_parameter("local_rescue_debug_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        image_msg_type = CompressedImage if self._topic_uses_compressed(image_topic) else Image
        self.create_subscription(image_msg_type, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(LocalRescueDebug, debug_topic, self.on_debug, qos_profile_sensor_data)
        self.create_subscription(ParameterEvent, "/parameter_events", self.on_parameter_event, 20)

        self.pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f"Sub image:       {image_topic}")
        self.get_logger().info(f"Sub rescue dbg:  {debug_topic}")
        self.get_logger().info("Sub param evt:   /parameter_events")
        self.get_logger().info(f"Pub debug image: {output_topic}")

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

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

    def _consume_publish_slot(self) -> bool:
        now_sec = self._now_sec()
        min_period = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 1e-3)
        with self.lock:
            if self.last_publish_sec > 0.0 and (now_sec - self.last_publish_sec) < min_period:
                return False
            self.last_publish_sec = now_sec
        return True

    @staticmethod
    def _param_value_to_text(param_value) -> str:
        t = int(param_value.type)
        if t == ParameterType.PARAMETER_BOOL:
            return str(bool(param_value.bool_value))
        if t == ParameterType.PARAMETER_INTEGER:
            return str(int(param_value.integer_value))
        if t == ParameterType.PARAMETER_DOUBLE:
            return f"{float(param_value.double_value):.6g}"
        if t == ParameterType.PARAMETER_STRING:
            return str(param_value.string_value)
        if t == ParameterType.PARAMETER_BYTE_ARRAY:
            return f"byte[{len(param_value.byte_array_value)}]"
        if t == ParameterType.PARAMETER_BOOL_ARRAY:
            return f"bool[{len(param_value.bool_array_value)}]"
        if t == ParameterType.PARAMETER_INTEGER_ARRAY:
            return f"int[{len(param_value.integer_array_value)}]"
        if t == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return f"double[{len(param_value.double_array_value)}]"
        if t == ParameterType.PARAMETER_STRING_ARRAY:
            return f"string[{len(param_value.string_array_value)}]"
        return "<unset>"

    def _prune_param_events_locked(self, now_sec: float):
        memory_sec = max(0.1, float(self.get_parameter("param_event_memory_sec").value))
        self.recent_param_events = [
            (ts, txt) for ts, txt in self.recent_param_events if (now_sec - ts) <= memory_sec
        ]

    def on_debug(self, msg: LocalRescueDebug):
        with self.lock:
            self.last_debug_msg = msg
            self.last_debug_recv_sec = self._now_sec()

    def on_parameter_event(self, msg: ParameterEvent):
        matcher_node_name = str(self.get_parameter("matcher_node_name").value).rstrip("/")
        event_node = str(msg.node).rstrip("/")
        if event_node != matcher_node_name:
            return

        updates = []
        for p in msg.new_parameters:
            updates.append(f"{p.name}={self._param_value_to_text(p.value)}")
        for p in msg.changed_parameters:
            updates.append(f"{p.name}={self._param_value_to_text(p.value)}")
        for p in msg.deleted_parameters:
            updates.append(f"{p.name}=<deleted>")

        if not updates:
            return

        now_sec = self._now_sec()
        with self.lock:
            for item in updates:
                self.recent_param_events.append((now_sec, item))
            self._prune_param_events_locked(now_sec)

        self.get_logger().info(
            f"matcher params changed: {', '.join(updates[:4])}{' ...' if len(updates) > 4 else ''}"
        )

    @staticmethod
    def _status_color(status: str) -> tuple[int, int, int]:
        if status == "RESCUED":
            return (0, 255, 0)
        if status == "AMBIGUOUS":
            return (0, 140, 255)
        if status == "ALL_FAIL_GATES":
            return (0, 0, 220)
        if status == "NO_CANDIDATES":
            return (0, 200, 255)
        if status.startswith("SKIP_"):
            return (255, 200, 0)
        return (200, 200, 200)

    def on_image(self, msg: Any):
        if not self._consume_publish_slot():
            return

        try:
            cv_img = self._decode_color_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"image decode failed: {exc}")
            return

        with self.lock:
            dbg = self.last_debug_msg
            dbg_age_sec = -1.0
            if self.last_debug_recv_sec > 0.0:
                dbg_age_sec = self._now_sec() - self.last_debug_recv_sec
            now_sec = self._now_sec()
            self._prune_param_events_locked(now_sec)
            param_events = [txt for _, txt in self.recent_param_events]

        self._draw_overlay(cv_img, dbg, dbg_age_sec, param_events)

        out = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)

    def _draw_overlay(
        self,
        cv_img: np.ndarray,
        dbg: Optional[LocalRescueDebug],
        dbg_age_sec: float,
        param_events: list[str],
    ):
        h, w = cv_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        focus_ref_id = int(self.get_parameter("focus_ref_id").value)
        max_refs = max(1, int(self.get_parameter("max_refs_to_draw").value))
        max_candidates = max(1, int(self.get_parameter("max_candidates_per_ref").value))
        show_scores = bool(self.get_parameter("show_candidate_scores").value)
        debug_timeout = max(1e-3, float(self.get_parameter("debug_timeout_sec").value))

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (8, 8), (630, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, cv_img, 0.55, 0, cv_img)

        y = 28
        cv2.putText(cv_img, "Local Rescue Debug (render-only)", (16, y), font, 0.55, (255, 255, 255), 1)
        y += 20

        if dbg is None:
            cv2.putText(cv_img, "No debug messages received yet.", (16, y), font, 0.52, (0, 200, 255), 1)
            self._draw_coordinate_axes(cv_img)
            return

        stale_dbg = dbg_age_sec < 0.0 or dbg_age_sec > debug_timeout
        cv2.putText(
            cv_img,
            (
                f"mode={dbg.mode} ready={dbg.context_ready} filtered_stale={dbg.filtered_stale} "
                f"debug_stale={stale_dbg}"
            ),
            (16, y),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            cv_img,
            (
                f"global={int(dbg.global_count)} final={int(dbg.final_count)} "
                f"attempts={int(dbg.local_attempts)} success={int(dbg.local_successes)} "
                f"applied={int(dbg.local_applied)}"
            ),
            (16, y),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            cv_img,
            (
                f"sim_floor={dbg.sim_floor:.2f} trace(P)={dbg.filter_trace:.2f} "
                f"amb: gap>={dbg.ambiguity_min_gap:.2f}, ratio>={dbg.ambiguity_min_ratio:.2f}"
            ),
            (16, y),
            font,
            0.47,
            (255, 255, 255),
            1,
        )
        y += 20
        cv2.putText(
            cv_img,
            f"entries={len(dbg.entries)} focus_ref_id={focus_ref_id}",
            (16, y),
            font,
            0.47,
            (255, 255, 255),
            1,
        )

        entries = list(dbg.entries)
        if focus_ref_id >= 0:
            entries = [e for e in entries if int(e.ref_id) == focus_ref_id]
        else:
            entries = entries[:max_refs]

        if not entries:
            cv2.putText(cv_img, "No entries to draw for current filter.", (16, 190), font, 0.55, (0, 200, 255), 2)
            self._draw_param_events(cv_img, param_events)
            self._draw_coordinate_axes(cv_img)
            return

        for entry in entries:
            px = int(round(float(entry.pred_x)))
            py = int(round(float(entry.pred_y)))
            radius_eff = max(1, int(round(float(entry.radius_eff_px))))

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(cv_img, (px, py), radius_eff, (0, 255, 255), 1)
                cv2.drawMarker(cv_img, (px, py), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            candidates = list(entry.candidates)[:max_candidates]
            for rank, cand in enumerate(candidates):
                cx = int(round(float(cand.x)))
                cy = int(round(float(cand.y)))
                color = (0, 220, 0) if rank == 0 else (220, 220, 0)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(cv_img, (cx, cy), 3, color, -1)
                    if show_scores:
                        cv2.putText(
                            cv_img,
                            f"s={float(cand.sim):.2f} q={float(cand.score):.2f}",
                            (cx + 3, cy - 3 - 10 * min(rank, 2)),
                            font,
                            0.32,
                            color,
                            1,
                        )

            status = str(entry.status)
            status_color = self._status_color(status)
            if float(entry.best_score) > 0.0:
                bx = int(round(float(entry.best_x)))
                by = int(round(float(entry.best_y)))
                if 0 <= px < w and 0 <= py < h and 0 <= bx < w and 0 <= by < h:
                    cv2.line(cv_img, (px, py), (bx, by), status_color, 1)
                if 0 <= bx < w and 0 <= by < h:
                    cv2.circle(cv_img, (bx, by), 6, status_color, 1)

            label_x = min(max(0, px + 6), max(0, w - 420))
            label_y = min(max(12, py + 12), max(12, h - 6))
            cv2.putText(
                cv_img,
                (
                    f"rid={int(entry.ref_id)} sig={float(entry.sigma_px):.2f} "
                    f"u={float(entry.uncertainty_u):.2f} r={float(entry.radius_eff_px):.1f} "
                    f"thr={float(entry.sim_threshold_eff):.2f} "
                    f"best={float(entry.best_sim):.2f}/{float(entry.best_score):.2f} {status}"
                ),
                (label_x, label_y),
                font,
                0.37,
                status_color,
                1,
            )

        self._draw_param_events(cv_img, param_events)
        self._draw_coordinate_axes(cv_img)

    def _draw_param_events(self, cv_img: np.ndarray, param_events: list[str]):
        if not bool(self.get_parameter("show_parameter_events").value):
            return
        if not param_events:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        h, _ = cv_img.shape[:2]

        show_items = param_events[-4:]
        box_h = 18 * (len(show_items) + 1) + 8
        y0 = max(0, h - box_h - 10)

        overlay = cv_img.copy()
        cv2.rectangle(overlay, (8, y0), (620, min(h - 1, y0 + box_h)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, cv_img, 0.55, 0, cv_img)

        y = y0 + 18
        cv2.putText(cv_img, "Recent matcher parameter changes:", (16, y), font, 0.45, (255, 255, 255), 1)
        for item in show_items:
            y += 16
            cv2.putText(cv_img, item, (16, y), font, 0.42, (180, 255, 180), 1)

    @staticmethod
    def _draw_coordinate_axes(cv_img):
        h, w = cv_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        margin = 10
        panel_w, panel_h = 80, 85
        panel_tl = (max(0, w - panel_w - margin), margin)
        panel_br = (min(w - 1, panel_tl[0] + panel_w), min(h - 1, panel_tl[1] + panel_h))

        origin = (panel_br[0] - 50, panel_br[1] - 18)
        axis_len = 30

        overlay = cv_img.copy()
        cv2.rectangle(overlay, panel_tl, panel_br, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, cv_img, 0.65, 0, cv_img)

        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)

        cv2.arrowedLine(
            cv_img,
            origin,
            (origin[0] + axis_len, origin[1]),
            color_x,
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            cv_img,
            "+x",
            (origin[0] + axis_len, origin[1] - 2),
            font,
            0.38,
            color_x,
            1,
            cv2.LINE_AA,
        )

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
            "+y",
            (origin[0] + 4, origin[1] - axis_len - 4),
            font,
            0.38,
            color_y,
            1,
            cv2.LINE_AA,
        )

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
            "+z",
            (z_center[0] - 20, z_center[1] + 12),
            font,
            0.38,
            color_z,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            cv_img,
            "Koordsystem",
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
