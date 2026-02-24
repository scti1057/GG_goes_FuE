from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
import numpy as np

@dataclass
class DetectionResult:
    # keypoints: (N,2) float32 in pixel coords (x,y)
    kpts_xy: np.ndarray
    # descriptors: (N,D) float32 or uint8 depending on detector
    desc: Optional[np.ndarray] = None
    # scores: (N,) float32 (if available)
    scores: Optional[np.ndarray] = None

class KeypointDetector(Protocol):
    name: str
    def detect_and_compute(self, gray: np.ndarray) -> DetectionResult:
        ...