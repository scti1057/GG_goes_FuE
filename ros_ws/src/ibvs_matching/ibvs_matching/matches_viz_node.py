import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from ibvs_msgs.msg import Matches

class MatchesVizNode(Node):
    def __init__(self):
        super().__init__('matches_viz_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('matches_topic', '/ibvs/matches')
        self.declare_parameter('output_topic', '/ibvs/debug/matches_image')
        self.declare_parameter('miss_max', 10)
        self.declare_parameter('radius', 3)

        self.bridge = CvBridge()
        self.last_matches = None  # type: Matches | None

        # ref_id -> (x,y,missed)
        self.tracks = {}

        img_topic = self.get_parameter('image_topic').value
        m_topic = self.get_parameter('matches_topic').value
        out_topic = self.get_parameter('output_topic').value

        self.sub_img = self.create_subscription(Image, img_topic, self.on_image, qos_profile_sensor_data)
        self.sub_m = self.create_subscription(Matches, m_topic, self.on_matches, qos_profile_sensor_data)

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(Image, out_topic, pub_qos)

        self.get_logger().info(f"Sub image:   {img_topic}")
        self.get_logger().info(f"Sub matches: {m_topic}")
        self.get_logger().info(f"Pub overlay: {out_topic}")

    def on_matches(self, msg: Matches):
        self.last_matches = msg

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"img convert failed: {e}")
            return

        miss_max = int(self.get_parameter('miss_max').value)
        radius = int(self.get_parameter('radius').value)

        # 1) missed++ for all known tracks
        for rid in list(self.tracks.keys()):
            x, y, missed = self.tracks[rid]
            missed += 1
            if missed > miss_max:
                del self.tracks[rid]
            else:
                self.tracks[rid] = (x, y, missed)

        # 2) apply latest matches (only reference points!)
        if self.last_matches is not None:
            m = self.last_matches
            xy = np.array(m.xy, dtype=np.float32)
            if xy.size % 2 == 0:
                pts = xy.reshape(-1, 2)
                for i, rid in enumerate(m.ref_id):
                    x, y = float(pts[i, 0]), float(pts[i, 1])
                    self.tracks[int(rid)] = (x, y, 0)  # reset missed

        # 3) draw tracks with fading (color intensity)
        out = bgr.copy()
        for rid, (x, y, missed) in self.tracks.items():
            intensity = max(0.0, 1.0 - (missed / float(miss_max)))
            g = int(255 * intensity)
            cv2.circle(out, (int(x), int(y)), radius, (0, g, 0), -1)

        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = MatchesVizNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
