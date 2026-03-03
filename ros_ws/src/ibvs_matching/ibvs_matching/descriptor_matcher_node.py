import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ibvs_msgs.msg import Keypoints, Matches


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # mat: (N,D)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (n + eps)


class DescriptorMatcherNode(Node):
    def __init__(self):
        super().__init__("descriptor_matcher_node")

        self.declare_parameter("keypoints_topic", "/ibvs/keypoints")
        self.declare_parameter("reference_topic", "/ibvs/reference/keypoints")
        self.declare_parameter("matches_topic", "/ibvs/matches")

        self.declare_parameter("match_threshold", 0.85)
        self.declare_parameter("mutual_check", True)

        self.ref_xy = None          # (K,2)
        self.ref_desc = None        # (K,D) normalized float32
        self.ref_D = 0

        kp_topic = self.get_parameter("keypoints_topic").value
        ref_topic = self.get_parameter("reference_topic").value
        out_topic = self.get_parameter("matches_topic").value

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        ref_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_ref = self.create_subscription(Keypoints, ref_topic, self.on_reference, ref_qos)
        self.sub_kp = self.create_subscription(Keypoints, kp_topic, self.on_keypoints, qos_profile_sensor_data)
        self.pub = self.create_publisher(Matches, out_topic, 10)

        self.get_logger().info(f"Sub keypoints: {kp_topic}")
        self.get_logger().info(f"Sub reference: {ref_topic}")
        self.get_logger().info(f"Pub matches:   {out_topic}")

    def on_reference(self, msg: Keypoints):
        xy = np.array(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn("Reference xy length not even; ignoring.")
            return
        k = xy.reshape(-1, 2)
        D = int(msg.descriptor_dim)

        if D <= 0:
            self.get_logger().warn("Reference has no descriptors (descriptor_dim=0); ignoring.")
            return

        if len(msg.descriptors) != k.shape[0] * D:
            self.get_logger().warn("Reference descriptors size mismatch; ignoring.")
            return

        desc = np.array(msg.descriptors, dtype=np.float32).reshape(k.shape[0], D)
        desc = l2_normalize(desc.astype(np.float32))

        self.ref_xy = k
        self.ref_desc = desc
        self.ref_D = D

        self.get_logger().info(f"Reference cached: K={k.shape[0]} D={D}")

    def on_keypoints(self, msg: Keypoints):
        if self.ref_desc is None:
            return  # wait until reference exists

        xy = np.array(msg.xy, dtype=np.float32)
        if xy.size % 2 != 0:
            self.get_logger().warn("Keypoints xy length not even; skipping frame.")
            return
        kpts = xy.reshape(-1, 2)
        n = kpts.shape[0]
        if n == 0:
            return

        D = int(msg.descriptor_dim)
        if D != self.ref_D or D <= 0:
            self.get_logger().warn(f"Descriptor dim mismatch: got {D}, ref {self.ref_D}.")
            return

        if len(msg.descriptors) != n * D:
            self.get_logger().warn("Keypoints descriptors size mismatch; skipping frame.")
            return

        desc = np.array(msg.descriptors, dtype=np.float32).reshape(n, D)
        desc = l2_normalize(desc.astype(np.float32))

        # cosine sim = dot product of normalized desc
        sim_mat = desc @ self.ref_desc.T   # (N,K)

        best_ref = np.argmax(sim_mat, axis=1)           # (N,)
        best_sim = sim_mat[np.arange(n), best_ref]      # (N,)

        thr = float(self.get_parameter("match_threshold").value)
        mutual = bool(self.get_parameter("mutual_check").value)

        keep = best_sim >= thr
        idx = np.where(keep)[0]
        if idx.size == 0:
            return

        if mutual:
            # each ref chooses best current; keep only mutual pairs
            best_cur_for_ref = np.argmax(sim_mat, axis=0)  # (K,)
            mutual_mask = np.array([best_cur_for_ref[best_ref[i]] == i for i in idx], dtype=bool)
            idx = idx[mutual_mask]
            if idx.size == 0:
                return

        out = Matches()
        out.header = msg.header

        out.ref_id = best_ref[idx].astype(np.uint32).tolist()

        xy_out = kpts[idx].astype(np.float32).reshape(-1)
        out.xy = xy_out.tolist()

        out.sim = best_sim[idx].astype(np.float32).tolist()

        self.pub.publish(out)


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
