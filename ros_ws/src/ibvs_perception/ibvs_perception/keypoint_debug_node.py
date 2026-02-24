import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from ibvs_msgs.msg import Keypoints
from cv_bridge import CvBridge
import cv2
import numpy as np

from ibvs_perception.detectors import create_detector

class KeypointDebugNode(Node):
    def __init__(self):
        super().__init__('keypoint_debug_node')

        # Params
        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_topic', '/ibvs/debug/keypoints_image')
        self.declare_parameter('detector_type', 'sift')   # sift|akaze|orb|superpoint|aliked|xfeat
        self.declare_parameter('device', 'cpu')           # cpu (später cuda möglich)
        self.declare_parameter('max_num_keypoints', 1024) # for superpoint/aliked
        self.declare_parameter('top_k', 1024)             # for xfeat
        self.declare_parameter('nfeatures', 800)          # for sift/orb

        self.bridge = CvBridge()

        self.detector_type = self.get_parameter('detector_type').value
        device = self.get_parameter('device').value
        max_kp = int(self.get_parameter('max_num_keypoints').value)
        top_k = int(self.get_parameter('top_k').value)
        nfeatures = int(self.get_parameter('nfeatures').value)

        # Create detector (simple kwargs mapping)
        kwargs = {}
        if self.detector_type in ('superpoint', 'aliked'):
            kwargs = {'max_num_keypoints': max_kp, 'device': device}
        elif self.detector_type == 'xfeat':
            kwargs = {'top_k': top_k, 'device': device}
        elif self.detector_type in ('sift', 'orb'):
            kwargs = {'nfeatures': nfeatures}

        self.det = create_detector(self.detector_type, **kwargs)
        self.get_logger().info(f"Using detector: {self.detector_type} ({self.det.__class__.__name__})")

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(Image, out_topic, pub_qos)
        self.kp_pub = self.create_publisher(Keypoints, '/ibvs/keypoints', 10)
        self.sub = self.create_subscription(Image, in_topic, self.cb, qos_profile_sensor_data)

        self.get_logger().info(f"Subscribing: {in_topic}")
        self.get_logger().info(f"Publishing:  {out_topic}")

    def cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            res = self.det.detect_and_compute(gray)
            kp_msg = Keypoints()
            kp_msg.header = msg.header

            # xy flatten
            kpts = res.kpts_xy
            kp_msg.xy = kpts.reshape(-1).astype(np.float32).tolist()

            # descriptors flatten (falls vorhanden)
            if res.desc is None:
                kp_msg.descriptor_dim = 0
                kp_msg.descriptors = []
            else:
                desc = res.desc
                if desc.dtype != np.float32:
                    desc = desc.astype(np.float32)
                kp_msg.descriptor_dim = int(desc.shape[1])
                kp_msg.descriptors = desc.reshape(-1).tolist()

            # scores optional
            if res.scores is None:
                kp_msg.scores = []
            else:
                kp_msg.scores = res.scores.astype(np.float32).reshape(-1).tolist()

            self.kp_pub.publish(kp_msg)
            kpts = res.kpts_xy

            overlay = bgr.copy()
            # draw small circles
            for x, y in kpts:
                cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)

            out = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            out.header = msg.header
            self.pub.publish(out)

            # occasional log
            self.get_logger().debug(f"kpts={kpts.shape[0]}")
        except Exception as e:
            self.get_logger().error(f"callback error: {e}")

def main():
    rclpy.init()
    node = KeypointDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
