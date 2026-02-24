import numpy as np
import torch
from lightglue import ALIKED
from .base import DetectionResult

class ALIKEDDetector:
    name = "aliked"

    def __init__(self, max_num_keypoints: int = 1024, device: str = "cpu"):
        self.device = device
        self.model = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)

    def detect_and_compute(self, gray: np.ndarray) -> DetectionResult:
        if gray.ndim != 2:
            raise ValueError("ALIKEDDetector expects a grayscale image (H,W).")

        img = gray.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0

        img3 = np.stack([img, img, img], axis=0)  # (3,H,W)
        t = torch.from_numpy(img3).to(self.device)

        with torch.no_grad():
            feats = self.model.extract(t)

        k = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        d = feats["descriptors"][0].detach().cpu().numpy().astype(np.float32)
        s = feats.get("keypoint_scores", feats.get("scores"))[0].detach().cpu().numpy().astype(np.float32)
        return DetectionResult(kpts_xy=k, desc=d, scores=s)
