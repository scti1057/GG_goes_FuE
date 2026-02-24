import numpy as np
import torch
from lightglue import SuperPoint
from .base import DetectionResult

class SuperPointDetector:
    name = "superpoint"

    def __init__(self, max_num_keypoints: int = 1024, device: str = "cpu"):
        self.device = device
        self.model = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)

    def detect_and_compute(self, gray: np.ndarray) -> DetectionResult:
        if gray.ndim != 2:
            raise ValueError("SuperPointDetector expects a grayscale image (H,W).")

        img = gray.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0

        # LightGlue expects (3,H,W) in [0,1]
        img3 = np.stack([img, img, img], axis=0)  # (3,H,W)
        t = torch.from_numpy(img3).to(self.device)

        with torch.no_grad():
            feats = self.model.extract(t)  # dict with batch dim

        k = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)       # (N,2)
        d = feats["descriptors"][0].detach().cpu().numpy().astype(np.float32)     # (N,D)
        s = feats.get("keypoint_scores", feats.get("scores"))[0].detach().cpu().numpy().astype(np.float32)          # (N,)
        return DetectionResult(kpts_xy=k, desc=d, scores=s)
