from __future__ import annotations

def create_detector(detector_type: str, **kwargs):
    t = detector_type.lower().strip()

    if t == "sift":
        from .sift import SIFTDetector
        return SIFTDetector(**kwargs)
    if t == "akaze":
        from .akaze import AKAZEDetector
        return AKAZEDetector(**kwargs)
    if t == "orb":
        from .orb import ORBDetector
        return ORBDetector(**kwargs)

    if t == "superpoint":
        from .superpoint import SuperPointDetector
        return SuperPointDetector(**kwargs)
    if t == "aliked":
        from .aliked import ALIKEDDetector
        return ALIKEDDetector(**kwargs)
    if t == "xfeat":
        from .xfeat import XFeatDetector
        return XFeatDetector(**kwargs)

    raise ValueError("Unknown detector_type. Supported: sift, akaze, orb, superpoint, aliked, xfeat")
