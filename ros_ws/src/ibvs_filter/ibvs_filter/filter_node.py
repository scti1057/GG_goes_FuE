#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import tf2_ros
import message_filters
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
        self.declare_parameter('z_depth', 0.5)
        self.declare_parameter('gate_threshold', 20.0)
        
        self.declare_parameter('base_frame', 'base_link') 
        self.declare_parameter('camera_frame', 'tool0') # camera_color_optical_frame
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        
        # Debug Flag
        self.declare_parameter('debug', True)

        self.filter_type = self.get_parameter('filter_type').value
        self.z_depth = self.get_parameter('z_depth').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.debug_mode = self.get_parameter('debug').value
        
        # --- Zustandsvariablen ---
        self.K = None
        self.filter = None
        self.reference_keypoints_raw = None 
        
        self.last_tf_stamp = None
        self.last_position = None
        self.last_rotation = None

        # CV Bridge für das Debug Bild
        self.cv_bridge = CvBridge()

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscriber (Asynchron) ---
        self.sub_cam_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.cam_info_callback, 10)
        self.sub_ref = self.create_subscription(Keypoints, '/ibvs/reference/keypoints', self.reference_callback, 10)
        
        # --- Publisher ---
        self.pub_filtered_points = self.create_publisher(Float32MultiArray, '/ibvs/filtered_features', 10)
        if self.debug_mode:
            self.pub_debug_img = self.create_publisher(Image, '/ibvs/filter_debug_image', 10)

        # --- Synchronisierte Subscriber (Matches + Image für Debug) ---
        self.sub_matches = message_filters.Subscriber(self, Matches, '/ibvs/matches')
        self.sub_image = message_filters.Subscriber(self, Image, self.get_parameter('image_topic').value)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_matches, self.sub_image], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

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
            self.get_parameter('q_noise').value, 
            self.get_parameter('r_noise').value, 
            self.get_parameter('gate_threshold').value
        )
        self.get_logger().info(f"Filter initialized, status: {self.filter.status}")

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
        self.get_logger().info(f"v_cam: {v_cam}")#, throttle_duration_sec=2.0)
        return v_cam, dt

    def sync_callback(self, matches_msg: Matches, img_msg: Image):
        """Wird aufgerufen, wenn Matches und das zugehörige Bild ankommen."""
        if self.filter is None:
            self.get_logger().warn("Warte auf CameraInfo...", throttle_duration_sec=2.0)
            return
        if self.reference_keypoints_raw is None:
            return

        # 1. Kinematik via TF holen
        v_ee, dt = self.get_camera_velocity_from_tf(matches_msg.header.stamp)
        if v_ee is None:
            return 

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

        # 3. Filter updaten oder blind prädizieren
        if valid_count < 4:
            self.filter.predict(v_ee, self.z_depth, dt)
        else:
            self.filter.predict(v_ee, self.z_depth, dt)
            self.filter.update(current_pixels, desired_pixels)

        # 4. Resultat publishen (ANSATZ B: Alle Referenzpunkte projizieren)
        if self.debug_mode:
            self.get_logger().info(f"Filter status: {self.filter.status}")
        if True: #self.filter.status in ["UPDATE", "INIT", "PREDICT", "REJECT (OUTLIER)", "REJECT (GEOMETRY)"]:
            all_reference_pts = self.reference_keypoints_raw.reshape(-1, 2).T
            
            # Die Magie: Wir filtern nicht die Features direkt, sondern nutzen die Homographie 
            # des Filters, um ALLE echten Referenzpunkte sauber in das aktuelle Bild zu projizieren!
            filtered_current_pts = self.filter.get_projected_points(all_reference_pts)
            
            out_msg = Float32MultiArray()
            out_msg.data = filtered_current_pts.T.flatten().tolist()
            self.pub_filtered_points.publish(out_msg)

            # 5. Debug Bild generieren und publishen
            if self.debug_mode:
                self.publish_debug_image(img_msg, desired_pixels, current_pixels)

    def publish_debug_image(self, img_msg, desired_pixels, current_pixels):
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Fehler: {e}")
            return

        # Wir zeichnen nur die aktuell gematchten Punkte, damit das Bild übersichtlich bleibt!
        if desired_pixels.shape[1] > 0:
            # Hole die projizierten (gefilterten) Koordinaten für diese spezifischen Matches
            proj_pixels = self.filter.get_projected_points(desired_pixels)

            for i in range(desired_pixels.shape[1]):
                pt_des = (int(desired_pixels[0, i]), int(desired_pixels[1, i]))
                pt_curr = (int(current_pixels[0, i]), int(current_pixels[1, i]))
                
                # 1. Soll-Position (Rot)
                cv2.circle(cv_img, pt_des, 4, (0, 0, 255), -1)
                # 2. Rohe Messung (Blau)
                cv2.circle(cv_img, pt_curr, 4, (255, 0, 0), -1)
                # 3. Matching Linie (Grün)
                cv2.line(cv_img, pt_des, pt_curr, (0, 255, 0), 1)

                # 4. Gefilterte Schätzung (Gelb)
                if proj_pixels.shape == desired_pixels.shape:
                    pt_proj = (int(proj_pixels[0, i]), int(proj_pixels[1, i]))
                    cv2.circle(cv_img, pt_proj, 5, (0, 255, 255), -1) 

        # --- UI Overlay (Texte und Boxen inkl. Legende) ---
        # 1. Hintergrundkasten vergrößern (jetzt bis Y=190 statt 75)
        cv2.rectangle(cv_img, (5, 5), (400, 190), (0, 0, 0, 0), -1)
        
        # 2. Bestehender Status-Text
        cv2.putText(cv_img, f"Filter: {self.filter_type.upper()}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        color = (0, 255, 0) if "UPDATE" in self.filter.status else (0, 0, 255)
        cv2.putText(cv_img, f"Status: {self.filter.status}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 3. Trennlinie zur Legende
        cv2.line(cv_img, (10, 75), (390, 75), (100, 100, 100), 1)

        # 4. Legenden-Einträge zeichnen
        # Soll-Position (Rot)
        cv2.circle(cv_img, (25, 100), 5, (0, 0, 255), -1)
        cv2.putText(cv_img, "Soll-Position (Referenz)", (45, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Rohe Messung (Blau)
        cv2.circle(cv_img, (25, 135), 5, (255, 0, 0), -1)
        cv2.putText(cv_img, "Rohe Messung (Kamera)", (45, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Gefilterte Schätzung (Gelb)
        cv2.circle(cv_img, (25, 170), 5, (0, 255, 255), -1)
        cv2.putText(cv_img, "Filter-Schaetzung", (45, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Publishen
        debug_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
        debug_msg.header = img_msg.header
        self.pub_debug_img.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()