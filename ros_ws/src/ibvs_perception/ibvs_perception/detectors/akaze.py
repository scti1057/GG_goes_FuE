import cv2
import numpy as np
from .base import DetectionResult

class AKAZEDetector:
    name = "akaze"

    def __init__(self):
        self.det = cv2.AKAZE_create()

    def detect_and_compute(self, gray: np.ndarray) -> DetectionResult:
        kpts, desc = self.det.detectAndCompute(gray, None)
        if not kpts:
            return DetectionResult(kpts_xy=np.zeros((0, 2), np.float32), desc=None, scores=None)

        xy = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32)
        scores = np.array([kp.response for kp in kpts], dtype=np.float32)
        # AKAZE descriptors are uint8 binary; keep as-is
        return DetectionResult(kpts_xy=xy, desc=desc, scores=scores)