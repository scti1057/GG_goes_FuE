#!/usr/bin/env python3

import threading
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from ibvs_msgs.msg import Keypoints, Matches
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, String, UInt32


class FilterDebugNode(Node):
    def __init__(self):
        super().__init__('ibvs_filter_debug_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('reference_topic', '/ibvs/reference/keypoints')
        self.declare_parameter('raw_matches_topic', '/ibvs/matches')
        self.declare_parameter('filtered_matches_topic', '/ibvs/filtered_features')
        self.declare_parameter('filter_status_topic', '/ibvs/filter/status')
        self.declare_parameter('filter_uncertainty_topic', '/ibvs/filter/uncertainty')
        self.declare_parameter('filter_update_status_topic', '/ibvs/filter/update_status')
        self.declare_parameter('filter_update_count_topic', '/ibvs/filter/update_count')
        self.declare_parameter(
            'filter_update_success_count_topic',
            '/ibvs/filter/update_success_count',
        )
        self.declare_parameter('active_count_topic', '/ibvs/filter/active_count')
        self.declare_parameter('debug_image_topic', '/ibvs/filter_debug_image')
        self.declare_parameter('debug_image_publish_rate', 30.0)
        self.declare_parameter('max_draw_points', 250)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.reference_topic = str(self.get_parameter('reference_topic').value)
        self.raw_matches_topic = str(self.get_parameter('raw_matches_topic').value)
        self.filtered_matches_topic = str(self.get_parameter('filtered_matches_topic').value)
        self.filter_status_topic = str(self.get_parameter('filter_status_topic').value)
        self.filter_uncertainty_topic = str(self.get_parameter('filter_uncertainty_topic').value)
        self.filter_update_status_topic = str(
            self.get_parameter('filter_update_status_topic').value
        )
        self.filter_update_count_topic = str(
            self.get_parameter('filter_update_count_topic').value
        )
        self.filter_update_success_count_topic = str(
            self.get_parameter('filter_update_success_count_topic').value
        )
        self.active_count_topic = str(self.get_parameter('active_count_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.debug_image_publish_rate = float(
            self.get_parameter('debug_image_publish_rate').value
        )
        self.max_draw_points = int(self.get_parameter('max_draw_points').value)

        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        self.reference_xy: Optional[np.ndarray] = None

        self.raw_ref_ids = np.zeros((0,), dtype=np.int64)
        self.raw_cur_xy = np.zeros((0, 2), dtype=np.float32)

        self.filtered_ref_ids = np.zeros((0,), dtype=np.int64)
        self.filtered_cur_xy = np.zeros((0, 2), dtype=np.float32)

        self.filter_status = 'UNKNOWN'
        self.filter_uncertainty = 0.0
        self.update_status = 'NO UPDATE YET'
        self.update_count = 0
        self.update_success_count = 0
        self.active_count = 0

        self.last_publish_time = None

        self.create_subscription(
            Keypoints,
            self.reference_topic,
            self.on_reference,
            10,
        )
        self.create_subscription(
            Matches,
            self.raw_matches_topic,
            self.on_raw_matches,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Matches,
            self.filtered_matches_topic,
            self.on_filtered_matches,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            self.filter_status_topic,
            self.on_filter_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32,
            self.filter_uncertainty_topic,
            self.on_filter_uncertainty,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            self.filter_update_status_topic,
            self.on_update_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            UInt32,
            self.filter_update_count_topic,
            self.on_update_count,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            UInt32,
            self.filter_update_success_count_topic,
            self.on_update_success_count,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            UInt32,
            self.active_count_topic,
            self.on_active_count,
            qos_profile_sensor_data,
        )

        image_msg_type = CompressedImage if self._topic_uses_compressed(self.image_topic) else Image
        self.create_subscription(
            image_msg_type,
            self.image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )

        self.pub_debug = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info(
            f'Debug node gestartet. image={self.image_topic}, out={self.debug_image_topic}'
        )

    @staticmethod
    def _topic_uses_compressed(topic: str) -> bool:
        return topic.endswith('/compressed') or topic.endswith('/compressedDepth')

    def _decode_color_bgr(self, msg: Any) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            try:
                bgr = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                bgr = None
            if bgr is None:
                bgr = cv2.imdecode(
                    np.frombuffer(bytes(msg.data), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            if bgr is None:
                raise RuntimeError('compressed image decode returned None')
            return bgr
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _consume_publish_slot(self) -> bool:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        min_period = 1.0 / max(self.debug_image_publish_rate, 1e-3)
        with self.lock:
            if self.last_publish_time is not None:
                if (now_sec - self.last_publish_time) < min_period:
                    return False
            self.last_publish_time = now_sec
        return True

    @staticmethod
    def _parse_matches(msg: Matches):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
        cur = xy.reshape(-1, 2)
        n = min(cur.shape[0], len(msg.ref_id))
        if n <= 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
        return np.asarray(msg.ref_id[:n], dtype=np.int64), cur[:n]

    def on_reference(self, msg: Keypoints):
        xy = np.asarray(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            return
        with self.lock:
            self.reference_xy = xy.reshape(-1, 2)

    def on_raw_matches(self, msg: Matches):
        ref_id, cur_xy = self._parse_matches(msg)
        with self.lock:
            self.raw_ref_ids = ref_id
            self.raw_cur_xy = cur_xy

    def on_filtered_matches(self, msg: Matches):
        ref_id, cur_xy = self._parse_matches(msg)
        with self.lock:
            self.filtered_ref_ids = ref_id
            self.filtered_cur_xy = cur_xy

    def on_filter_status(self, msg: String):
        with self.lock:
            self.filter_status = str(msg.data)

    def on_filter_uncertainty(self, msg: Float32):
        with self.lock:
            self.filter_uncertainty = float(msg.data)

    def on_update_status(self, msg: String):
        with self.lock:
            self.update_status = str(msg.data)

    def on_update_count(self, msg: UInt32):
        with self.lock:
            self.update_count = int(msg.data)

    def on_update_success_count(self, msg: UInt32):
        with self.lock:
            self.update_success_count = int(msg.data)

    def on_active_count(self, msg: UInt32):
        with self.lock:
            self.active_count = int(msg.data)

    def on_image(self, img_msg: Any):
        if not self._consume_publish_slot():
            return

        try:
            cv_img = self._decode_color_bgr(img_msg)
        except Exception as exc:
            self.get_logger().warn(f'image decode failed: {exc}')
            return

        with self.lock:
            reference_xy = None if self.reference_xy is None else self.reference_xy.copy()
            raw_ref_ids = self.raw_ref_ids.copy()
            raw_cur_xy = self.raw_cur_xy.copy()
            filtered_ref_ids = self.filtered_ref_ids.copy()
            filtered_cur_xy = self.filtered_cur_xy.copy()
            filter_status = self.filter_status
            filter_uncertainty = self.filter_uncertainty
            update_status = self.update_status
            update_count = self.update_count
            update_success_count = self.update_success_count
            active_count = self.active_count

        self._draw_overlay(
            cv_img,
            reference_xy,
            raw_ref_ids,
            raw_cur_xy,
            filtered_ref_ids,
            filtered_cur_xy,
            filter_status,
            filter_uncertainty,
            update_status,
            update_count,
            update_success_count,
            active_count,
        )

        out_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, 'bgr8')
        out_msg.header = img_msg.header
        self.pub_debug.publish(out_msg)

    def _draw_overlay(
        self,
        cv_img,
        reference_xy,
        raw_ref_ids,
        raw_cur_xy,
        filtered_ref_ids,
        filtered_cur_xy,
        filter_status,
        filter_uncertainty,
        update_status,
        update_count,
        update_success_count,
        active_count,
    ):
        max_draw = max(1, int(self.max_draw_points))

        if reference_xy is not None and reference_xy.shape[0] > 0:
            for i in range(min(raw_ref_ids.size, raw_cur_xy.shape[0], max_draw)):
                rid = int(raw_ref_ids[i])
                if rid < 0 or rid >= reference_xy.shape[0]:
                    continue
                p_ref = tuple(np.round(reference_xy[rid]).astype(np.int32).tolist())
                p_raw = tuple(np.round(raw_cur_xy[i]).astype(np.int32).tolist())
                cv2.circle(cv_img, p_ref, 2, (0, 0, 255), -1)
                cv2.circle(cv_img, p_raw, 2, (255, 0, 0), -1)

        for i in range(min(filtered_ref_ids.size, filtered_cur_xy.shape[0], max_draw)):
            p_f = tuple(np.round(filtered_cur_xy[i]).astype(np.int32).tolist())
            cv2.circle(cv_img, p_f, 2, (0, 255, 255), -1)

        filter_name = 'UNKNOWN'
        status_name = filter_status
        if '|' in filter_status:
            left, right = filter_status.split('|', 1)
            filter_name = left.strip()
            status_name = right.strip()

        overlay = cv_img.copy()
        box_w, box_h = 370, 285
        cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, cv_img, 0.6, 0, cv_img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        x_txt, y_txt = 15, 25
        line_h = 20

        # cv2.putText(cv_img, 'Filter Debug (separate node)', (x_txt, y_txt), font, 0.5, (255, 255, 255), 1)
        # y_txt += line_h
        cv2.putText(cv_img, f'Filter: {filter_name} | Status: {status_name}', (x_txt, y_txt), font, 0.5, (255, 255, 255), 1)
        y_txt += line_h
        # cv2.putText(cv_img, f'Status: {status_name}', (x_txt, y_txt), font, 0.5, (255, 255, 255), 1)
        # y_txt += line_h
        cv2.putText(
            cv_img,
            f'Unsicherheit (trace P): {filter_uncertainty:.2f}',
            (x_txt, y_txt),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y_txt += line_h
        cv2.putText(cv_img, f'Raw matches: {raw_ref_ids.size}', (x_txt, y_txt), font, 0.5, (255, 255, 255), 1)
        y_txt += line_h
        cv2.putText(
            cv_img,
            f'Filtered points: {filtered_ref_ids.size}',
            (x_txt, y_txt),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y_txt += line_h
        cv2.putText(cv_img, f'Update-Status: {update_status}', (x_txt, y_txt), font, 0.5, (255, 255, 255), 1)
        y_txt += line_h
        cv2.putText(
            cv_img,
            f'Updates: {update_count} (ok: {update_success_count})',
            (x_txt, y_txt),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y_txt += line_h
        cv2.putText(
            cv_img,
            f'Aktiv gefilterte Keypoints: {active_count}',
            (x_txt, y_txt),
            font,
            0.5,
            (255, 255, 255),
            1,
        )
        y_txt += line_h + 4
        cv2.putText(cv_img, 'Legende:', (x_txt, y_txt), font, 0.5, (220, 220, 220), 1)
        y_txt += line_h
        cv2.circle(cv_img, (x_txt + 8, y_txt - 4), 4, (0, 0, 255), -1)
        cv2.putText(cv_img, 'Rot: Referenzpunkt', (x_txt + 20, y_txt), font, 0.45, (255, 255, 255), 1)
        y_txt += line_h
        cv2.circle(cv_img, (x_txt + 8, y_txt - 4), 4, (255, 0, 0), -1)
        cv2.putText(cv_img, 'Blau: Roh-Match', (x_txt + 20, y_txt), font, 0.45, (255, 255, 255), 1)
        y_txt += line_h
        cv2.circle(cv_img, (x_txt + 8, y_txt - 4), 4, (0, 255, 255), -1)
        cv2.putText(cv_img, 'Gelb: Gefilterter Punkt', (x_txt + 20, y_txt), font, 0.45, (255, 255, 255), 1)
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


def main(args=None):
    rclpy.init(args=args)
    node = FilterDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
