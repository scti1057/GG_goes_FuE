import numpy as np
import torch
import sys
import os
from pathlib import Path
from .base import DetectionResult

# accelerated_features expects torch.inference_mode (newer torch versions).
# Provide a no_grad fallback for environments shipping torch<1.9.
if not hasattr(torch, "inference_mode"):
    torch.inference_mode = torch.no_grad  # type: ignore[attr-defined]

class XFeatDetector:
    name = "xfeat"

    def __init__(
        self,
        top_k: int = 1024,
        device: str = "cpu",
        repo_dir: str | None = None,
    ):
        self.top_k = top_k
        self.device = device

        if repo_dir is None:
            repo_dir = os.environ.get(
                "IBVS_XFEAT_REPO_DIR",
                "/home/ros_ws/third_party/accelerated_features",
            )

        repo_path = Path(repo_dir)
        if not repo_path.exists():
            raise FileNotFoundError(f"accelerated_features repo not found at: {repo_dir}")

        # make 'modules.xfeat' importable
        sys.path.append(str(repo_path))
        from modules.xfeat import XFeat  # type: ignore

        self.model = XFeat().to(device).eval()

    def detect_and_compute(self, gray: np.ndarray) -> DetectionResult:
        if gray.ndim != 2:
            raise ValueError("XFeatDetector expects a grayscale image (H,W).")

        img = gray.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0

        img3 = np.stack([img, img, img], axis=0)          # (3,H,W)
        x = torch.from_numpy(img3).unsqueeze(0).to(self.device)  # (1,3,H,W)

        with torch.no_grad():
            out = self.model.detectAndCompute(x, top_k=self.top_k)[0]

        k = out["keypoints"].detach().cpu().numpy().astype(np.float32)     # (N,2)
        d = out["descriptors"].detach().cpu().numpy().astype(np.float32)   # (N,D)
        s = out["scores"].detach().cpu().numpy().astype(np.float32)        # (N,)
        return DetectionResult(kpts_xy=k, desc=d, scores=s)
