import numpy as np
import cv2

class BaseFilter:
    """
    Zentrale Basisklasse für alle Image-Space Filter.
    Übernimmt das RANSAC-Matching, die dynamische Bounding-Box und den Geometrie-Check.
    """
    def __init__(self, K):
        self.K = K
        self.initialized = False
        self.status = "INIT"
        self.proxy_ref = None 
        self.H_filtered = np.eye(3)

    def reset(self):
        self.initialized = False
        self.status = "INIT"
        self.proxy_ref = None
        self.H_filtered = np.eye(3)

    def force_relocalization(self):
        self.initialized = False
        self.status = "RELOCALIZING"

    def _init_dynamic_proxies(self, desired_pixels):
        u_min, u_max = np.min(desired_pixels[0, :]), np.max(desired_pixels[0, :])
        v_min, v_max = np.min(desired_pixels[1, :]), np.max(desired_pixels[1, :])
        pad = 10.0
        self.proxy_ref = np.array([
            [u_min - pad, v_min - pad],
            [u_max + pad, v_min - pad],
            [u_max + pad, v_max + pad],
            [u_min - pad, v_max + pad]
        ], dtype=np.float32)

    def _get_raw_measurement(self, current_pixels, desired_pixels):
        if current_pixels is None or current_pixels.shape[1] < 4:
            return None, None, "MISSING"

        src_pts = desired_pixels.T.astype(np.float32)
        dst_pts = current_pixels.T.astype(np.float32)
        H_raw, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H_raw is None:
            return None, None, "REJECT (RANSAC FAIL)"

        if self.proxy_ref is None:
            self._init_dynamic_proxies(desired_pixels)

        proxy_hom = np.hstack((self.proxy_ref, np.ones((4, 1))))
        z_raw_hom = (H_raw @ proxy_hom.T).T
        
        z_k = np.zeros((8, 1))
        for i in range(4):
            z_k[i*2, 0] = z_raw_hom[i, 0] / z_raw_hom[i, 2]
            z_k[i*2 + 1, 0] = z_raw_hom[i, 1] / z_raw_hom[i, 2]
            
        return z_k, H_raw, self.status

    def _check_geometry_and_update_H(self, state_8d, log_on_fail=True):
        proxy_filtered = state_8d.reshape(4, 2).astype(np.float32)
        
        def cross_z(A, B, C):
            return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
        
        c1 = cross_z(proxy_filtered[0], proxy_filtered[1], proxy_filtered[2])
        c2 = cross_z(proxy_filtered[1], proxy_filtered[2], proxy_filtered[3])
        c3 = cross_z(proxy_filtered[2], proxy_filtered[3], proxy_filtered[0])
        c4 = cross_z(proxy_filtered[3], proxy_filtered[0], proxy_filtered[1])
        
        signs = [c > 0 for c in [c1, c2, c3, c4]]
        is_convex = all(signs) or not any(signs)
        
        if not is_convex:
            if log_on_fail:
                print("\n[Filter WARNING] Geometry check failed (Topology Flip).")
                print(("[Filter WARNING] Coordinates of the 4 corners:"))
                print(proxy_filtered)
                print(20*"-")
            return False
            
        self.H_filtered = cv2.getPerspectiveTransform(self.proxy_ref, proxy_filtered)
        return True

    def update_geometry_from_state(self, state_8d, log_on_fail=False):
        """Update H_filtered from a predicted 8D corner state."""
        if self.proxy_ref is None:
            return False
        state_8d = np.asarray(state_8d, dtype=np.float64).reshape(8, 1)
        return self._check_geometry_and_update_H(state_8d, log_on_fail=log_on_fail)

    def get_projected_points(self, desired_features):
        if not self.initialized or self.H_filtered is None:
            return desired_features
        M = desired_features.shape[1]
        hom_pts = np.vstack((desired_features, np.ones((1, M))))
        proj_hom = self.H_filtered @ hom_pts
        return np.vstack((proj_hom[0, :] / proj_hom[2, :], proj_hom[1, :] / proj_hom[2, :]))

    def get_proxy_corners(self):
        """Returns (proxy_ref_4x2, proxy_est_4x2_or_None)."""
        if self.proxy_ref is None:
            return None, None

        proxy_ref = self.proxy_ref.copy()
        proxy_est = None
        if self.initialized and self.H_filtered is not None:
            proxy_est = self.get_projected_points(proxy_ref.T).T

        return proxy_ref, proxy_est

    def _transform_twist_ee_to_cam(self, v_ee):
        """Map project-specific EE twist components into the camera frame.

        The current convention used by the IBVS filters is:
        [vx, vy, vz, wx, wy, wz] -> [-vx, +vy, +vz, -wx, -wy, -wz]
        """
        v_ee = np.asarray(v_ee, dtype=np.float64).reshape(6,)
        return np.array([
            -v_ee[0],
             v_ee[1],
             v_ee[2],
             v_ee[3],
             v_ee[4],
            -v_ee[5],
        ], dtype=np.float64)

    def predict(self, v_ee, Z_est, dt): raise NotImplementedError
    def update(self, current_pixels, desired_pixels, ref_ids=None): raise NotImplementedError
