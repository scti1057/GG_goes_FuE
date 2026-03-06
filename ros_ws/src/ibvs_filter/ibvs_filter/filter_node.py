#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import threading

import tf2_ros
import message_filters
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo, Image
from ibvs_msgs.msg import Matches, Keypoints
from cv_bridge import CvBridge

from ibvs_filter.core.ekf import ExtendedKalmanFilter
from ibvs_filter.core.ukf import UnscentedKalmanFilter
from ibvs_filter.core.eskf import ErrorStateKalmanFilter
from ibvs_filter.core.skf import StandardKalmanFilter

class FilterNode(Node):
    def __init__(self):
        super().__init__('ibvs_filter_node')

        # --- Parameter ---
        self.declare_parameter('filter_type', 'ekf')
        self.declare_parameter('q_noise', 1.0)
        self.declare_parameter('r_noise', 50.0)
        self.declare_parameter('z_depth', 0.25)
        self.declare_parameter('gate_threshold', 20.0)
        self.declare_parameter('predict_rate', 60.0) # Hz (Reduziert für Performance, 60Hz reicht völlig)
        
        self.declare_parameter('base_frame', 'base_link') 
        self.declare_parameter('camera_frame', 'camera_color_optical_frame') # camera_color_optical_frame
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        
        # Debug Flag
        self.declare_parameter('debug', True)

        self.filter_type = self.get_parameter('filter_type').value
        self.q_noise = self.get_parameter('q_noise').value
        self.r_noise = self.get_parameter('r_noise').value
        self.gate_threshold = self.get_parameter('gate_threshold').value
        self.z_depth = self.get_parameter('z_depth').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.debug_mode = self.get_parameter('debug').value
        self.predict_rate = self.get_parameter('predict_rate').value
        
        # --- Zustandsvariablen ---
        self.K = None
        self.filter = None
        self.reference_keypoints_raw = None 
        
        # Thread-Safety: Lock für den Filter-Zustand (da Predict/Update nun parallel laufen können)
        self.lock = threading.Lock()
        
        # Caching für Debug-Visualisierung (da Update und Image nun asynchron sind)
        self.last_current_pixels = None
        self.last_desired_pixels = None
        
        self.last_tf_stamp = None
        self.last_position = None
        self.last_rotation = None

        # CV Bridge für das Debug Bild
        self.cv_bridge = CvBridge()

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Callback Group für Multi-Threading ---
        # Erlaubt, dass Timer und Subscriber parallel verarbeitet werden
        self.cb_group = ReentrantCallbackGroup()

        # --- Subscriber (Asynchron) ---
        self.sub_cam_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.cam_info_callback, 10, callback_group=self.cb_group)
        self.sub_ref = self.create_subscription(Keypoints, '/ibvs/reference/keypoints', self.reference_callback, 10, callback_group=self.cb_group)
        
        # --- Publisher ---
        self.pub_filtered_points = self.create_publisher(Matches, '/ibvs/filtered_features', 10)
        if self.debug_mode:
            self.pub_debug_img = self.create_publisher(Image, '/ibvs/filter_debug_image', 10)

        # --- Asynchrone Subscriber & Timer ---
        # 1. Update Schritt (Event-basiert bei neuen Matches)
        self.sub_matches = self.create_subscription(Matches, '/ibvs/matches', self.matches_callback, 10, callback_group=self.cb_group)
        
        # 2. Debug Image (Event-basiert bei neuem Bild)
        self.sub_image = self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, 10, callback_group=self.cb_group)

        # 3. Predict Schritt (Zeit-basiert, fester Takt)
        self.timer = self.create_timer(1.0 / self.predict_rate, self.timer_callback, callback_group=self.cb_group)

        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.get_logger().info(f"Filter Node gestartet. Modus: {self.filter_type}. Warte auf K-Matrix und Referenz...")

    def cam_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.init_filter()
            self.get_logger().info("CameraInfo empfangen!")

    def init_filter(self):
        if self.filter_type == 'ekf': self.filter = ExtendedKalmanFilter(self.K)
        elif self.filter_type == 'ukf': self.filter = UnscentedKalmanFilter(self.K)
        elif self.filter_type == 'eskf': self.filter = ErrorStateKalmanFilter(self.K)
        elif self.filter_type == 'skf': self.filter = StandardKalmanFilter(self.K)
        else:
            self.get_logger().error(f"Unbekannter Filtertyp: {self.filter_type}")
            return
            
        self.filter.set_Q_R_gate(
            self.q_noise,
            self.r_noise,
            self.gate_threshold
        )
        self.get_logger().info(f"Filter initialized, status: {self.filter.status}")

    def _on_parameters_changed(self, params):
        next_q = self.q_noise
        next_r = self.r_noise
        next_gate = self.gate_threshold
        next_z = self.z_depth

        for p in params:
            if p.name == 'filter_type':
                return SetParametersResult(
                    successful=False,
                    reason='filter_type cannot be changed at runtime. Restart node.'
                )
            if p.name == 'q_noise':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='q_noise must be > 0')
                next_q = float(p.value)
            elif p.name == 'r_noise':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='r_noise must be > 0')
                next_r = float(p.value)
            elif p.name == 'gate_threshold':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='gate_threshold must be > 0')
                next_gate = float(p.value)
            elif p.name == 'z_depth':
                if p.value <= 0.0:
                    return SetParametersResult(successful=False, reason='z_depth must be > 0')
                next_z = float(p.value)

        self.q_noise = next_q
        self.r_noise = next_r
        self.gate_threshold = next_gate
        self.z_depth = next_z

        if self.filter is not None:
            self.filter.set_Q_R_gate(self.q_noise, self.r_noise, self.gate_threshold)
            self.get_logger().info(
                f"Tuning updated: q_noise={self.q_noise:.4f}, "
                f"r_noise={self.r_noise:.4f}, gate_threshold={self.gate_threshold:.4f}, "
                f"z_depth={self.z_depth:.4f}"
            )

        return SetParametersResult(successful=True)

    def reference_callback(self, msg: Keypoints):
        if self.reference_keypoints_raw is None:
            self.reference_keypoints_raw = np.array(msg.xy)
            self.get_logger().info(f"Neue Referenz empfangen! ({len(msg.xy)//2} Keypoints)")
            if self.filter is not None:
                self.filter.force_relocalization()

    def get_camera_velocity_from_tf(self, time_stamp):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, time_stamp, rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f"TF Fehler: {e}", throttle_duration_sec=2.0)
            return None, 0.0

        current_time = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        pos = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        rot = R.from_quat([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

        v_cam = np.zeros(6)
        dt = 0.033

        if self.last_position is not None:
            dt = current_time - self.last_tf_stamp
            if dt > 0.001:
                dp_base = pos - self.last_position
                v_cam[:3] = rot.inv().apply(dp_base) / dt
                r_rel = self.last_rotation.inv() * rot
                v_cam[3:] = r_rel.as_rotvec() / dt

        self.last_position = pos
        self.last_rotation = rot
        self.last_tf_stamp = current_time

        if False: #self.debug_mode:
            self.get_logger().info(f"Transform: {t}")
            self.get_logger().info(f"v_cam: {v_cam}")#, throttle_duration_sec=2.0)
        return v_cam, dt

    def timer_callback(self):
        """1. Predict Schritt: Wird mit fester Frequenz ausgeführt."""
        if self.filter is None:
            return
        if self.reference_keypoints_raw is None:
            return

        # Kinematik via TF holen (zum aktuellen Zeitpunkt)
        # Wir nutzen Time() -> 0, um den aktuellsten Transform zu bekommen
        v_ee, dt = self.get_camera_velocity_from_tf(rclpy.time.Time())
        
        if v_ee is None:
            return

        # Predict ausführen
        with self.lock:
            self.filter.predict(v_ee, self.z_depth, dt)
            
            # Resultat publishen (ANSATZ B: Alle Referenzpunkte projizieren)
            # Wir nutzen den Lock auch hier, damit sich der Zustand während der Projektion nicht ändert
            all_reference_pts = self.reference_keypoints_raw.reshape(-1, 2).T
            filtered_current_pts = self.filter.get_projected_points(all_reference_pts)

        # Nachricht im Matches-Format bauen
        out_msg = Matches()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.camera_frame
        
        num_pts = filtered_current_pts.shape[1]
        # Die ID entspricht dem Index im Referenz-Array. Wir stellen sicher, dass es ints sind.
        out_msg.ref_id = [int(i) for i in range(num_pts)] 
        out_msg.xy = filtered_current_pts.T.flatten().tolist()
        out_msg.sim = [1.0] * num_pts # Da vom Filter generiert, setzen wir Confidence auf 1.0
        
        self.pub_filtered_points.publish(out_msg)

    def matches_callback(self, matches_msg: Matches):
        """2. Update Schritt: Wird ausgeführt, wenn neue Matches da sind."""
        if self.filter is None or self.reference_keypoints_raw is None: return

        # 2. Matches extrahieren und zuordnen
        num_matches = len(matches_msg.ref_id)
        current_pixels = np.zeros((2, num_matches))
        desired_pixels = np.zeros((2, num_matches))
        
        valid_count = 0
        for i, ref_idx in enumerate(matches_msg.ref_id):
            if (ref_idx * 2 + 1) < len(self.reference_keypoints_raw):
                current_pixels[0, valid_count] = matches_msg.xy[i * 2]
                current_pixels[1, valid_count] = matches_msg.xy[i * 2 + 1]
                desired_pixels[0, valid_count] = self.reference_keypoints_raw[ref_idx * 2]
                desired_pixels[1, valid_count] = self.reference_keypoints_raw[ref_idx * 2 + 1]
                valid_count += 1

        current_pixels = current_pixels[:, :valid_count]
        desired_pixels = desired_pixels[:, :valid_count]

        # Caching für Debugging
        self.last_current_pixels = current_pixels
        self.last_desired_pixels = desired_pixels

        # Update ausführen (nur wenn genug Matches da sind)
        if valid_count >= 4:
            with self.lock:
                self.filter.update(current_pixels, desired_pixels)

    def image_callback(self, img_msg: Image):
        """3. Debug Schritt: Zeichnet Overlay, wenn ein Bild kommt."""
        if not self.debug_mode or self.filter is None:
            return
            
        # Wir nutzen die gecacheten Matches vom letzten Update-Schritt
        if self.last_desired_pixels is None or self.last_current_pixels is None:
            return

        if self.debug_mode:
            # self.get_logger().info(f"Filter status: {self.filter.status}")
            self.publish_debug_image(img_msg, self.last_desired_pixels, self.last_current_pixels)

    def publish_debug_image(self, img_msg, desired_pixels, current_pixels):
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Fehler: {e}")
            return

        # Filter-Infos holen (Thread-Safe)
        with self.lock:
            p_trace = np.trace(self.filter.P) if hasattr(self.filter, 'P') else 0.0
            filter_status = self.filter.status
            proxy_ref, proxy_est = self.filter.get_proxy_corners()

        # Wir zeichnen nur die aktuell gematchten Punkte, damit das Bild übersichtlich bleibt!
        if desired_pixels.shape[1] > 0:
            # Hole die projizierten (gefilterten) Koordinaten für diese spezifischen Matches
            # Thread-Safe Zugriff auf den Filter
            with self.lock:
                proj_pixels = self.filter.get_projected_points(desired_pixels)

            for i in range(desired_pixels.shape[1]):
                pt_des = (int(desired_pixels[0, i]), int(desired_pixels[1, i]))
                pt_curr = (int(current_pixels[0, i]), int(current_pixels[1, i]))
                
                # 1. Soll-Position (Rot)
                # cv2.circle(cv_img, pt_des, 2, (0, 0, 255), -1)
                # 2. Rohe Messung (Blau)
                cv2.circle(cv_img, pt_curr, 2, (255, 0, 0), -1)
                # 3. Matching Linie (Grün)
                # cv2.line(cv_img, pt_des, pt_curr, (0, 255, 0), 1)

                # 4. Gefilterte Schätzung (Gelb)
                if proj_pixels.shape == desired_pixels.shape:
                    pt_proj = (int(proj_pixels[0, i]), int(proj_pixels[1, i]))
                    cv2.circle(cv_img, pt_proj, 2, (0, 255, 255), -1) 

        # Proxy-Ecken visualisieren
        if proxy_ref is not None and np.isfinite(proxy_ref).all():
            pts_ref = np.round(proxy_ref).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cv_img, [pts_ref], True, (0, 128, 255), 1, cv2.LINE_AA)
            for i, p in enumerate(proxy_ref):
                c = (int(p[0]), int(p[1]))
                cv2.circle(cv_img, c, 4, (0, 128, 255), -1)
                cv2.putText(cv_img, f"R{i}", (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1)

        if proxy_est is not None and np.isfinite(proxy_est).all():
            pts_est = np.round(proxy_est).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cv_img, [pts_est], True, (255, 255, 0), 1, cv2.LINE_AA)
            for i, p in enumerate(proxy_est):
                c = (int(p[0]), int(p[1]))
                cv2.circle(cv_img, c, 4, (255, 255, 0), -1)
                cv2.putText(cv_img, f"E{i}", (c[0] + 4, c[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # --- UI Overlay (Texte und Boxen inkl. Legende) ---
        # Transparentes Overlay
        overlay = cv_img.copy()
        box_w, box_h = 260, 210
        cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
        
        alpha = 0.4 # Transparenz (40% Schwarz, 60% Bild)
        cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0, cv_img)
        
        # Text Setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        white = (255, 255, 255)
        x_txt, y_txt = 15, 25
        line_h = 20
        
        # 1. Info Block
        cv2.putText(cv_img, f"Filter: {self.filter_type.upper()}", (x_txt, y_txt), font, font_scale, white, thickness)
        y_txt += line_h
        
        color_status = (0, 255, 0) if "UPDATE" in filter_status else (0, 0, 255)
        cv2.putText(cv_img, f"Status: {filter_status}", (x_txt, y_txt), font, font_scale, color_status, thickness)
        y_txt += line_h

        cv2.putText(cv_img, f"Matches: {desired_pixels.shape[1]}", (x_txt, y_txt), font, font_scale, white, thickness)
        y_txt += line_h
        
        cv2.putText(cv_img, f"Uncertainty: {p_trace:.2f}", (x_txt, y_txt), font, font_scale, white, thickness)
        y_txt += 10

        # 2. Trennlinie
        cv2.line(cv_img, (10, y_txt), (box_w, y_txt), (150, 150, 150), 1)
        y_txt += 20

        # 3. Legende
        # Soll (Rot)
        cv2.circle(cv_img, (25, y_txt-5), 3, (0, 0, 255), -1)
        cv2.putText(cv_img, "Ref (Soll)", (40, y_txt), font, font_scale, white, thickness)
        y_txt += line_h

        # Ist (Blau)
        cv2.circle(cv_img, (25, y_txt-5), 3, (255, 0, 0), -1)
        cv2.putText(cv_img, "Meas (Ist)", (40, y_txt), font, font_scale, white, thickness)
        y_txt += line_h

        # Filter (Gelb)
        cv2.circle(cv_img, (25, y_txt-5), 3, (0, 255, 255), -1)
        cv2.putText(cv_img, "Est (Filter)", (40, y_txt), font, font_scale, white, thickness)
        y_txt += line_h

        # Proxy Ref (Orange)
        cv2.circle(cv_img, (25, y_txt-5), 3, (0, 128, 255), -1)
        cv2.putText(cv_img, "Proxy Ref", (40, y_txt), font, font_scale, white, thickness)
        y_txt += line_h

        # Proxy Est (Cyan)
        cv2.circle(cv_img, (25, y_txt-5), 3, (255, 255, 0), -1)
        cv2.putText(cv_img, "Proxy Est", (40, y_txt), font, font_scale, white, thickness)

        # Publishen
        debug_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
        debug_msg.header = img_msg.header
        self.pub_debug_img.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FilterNode()
    # MultiThreadedExecutor verhindert, dass die Bildverarbeitung den Filter blockiert
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
