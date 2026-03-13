# modelfilter_ibvs/core/filters/ekf.py
import numpy as np
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class ExtendedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        self.x = np.zeros((0, 1), dtype=np.float64)
        self.P = np.zeros((0, 0), dtype=np.float64)
        self.H = np.zeros((0, 0), dtype=np.float64)
        self.q_noise = 1.0
        self.r_noise = 50.0
        self.Q = np.zeros((0, 0), dtype=np.float64)
        self.R = np.zeros((0, 0), dtype=np.float64)
        self.gate_thresh = 20.0

    def _set_state_dim(self, active_count):
        l_dim = int(2 * active_count)
        if l_dim <= 0:
            self.x = np.zeros((0, 1), dtype=np.float64)
            self.P = np.zeros((0, 0), dtype=np.float64)
            self.H = np.zeros((0, 0), dtype=np.float64)
            self.Q = np.zeros((0, 0), dtype=np.float64)
            self.R = np.zeros((0, 0), dtype=np.float64)
            return
        if self.x.shape[0] == l_dim:
            return
        self.x = np.zeros((l_dim, 1), dtype=np.float64)
        self.P = np.eye(l_dim, dtype=np.float64) * 1000.0
        self.H = np.eye(l_dim, dtype=np.float64)
        self.Q = np.eye(l_dim, dtype=np.float64) * self.q_noise
        self.R = np.eye(l_dim, dtype=np.float64) * self.r_noise

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.q_noise = float(q_val)
        self.r_noise = float(r_val)
        l_dim = self.x.shape[0]
        self.Q = np.eye(l_dim, dtype=np.float64) * self.q_noise
        self.R = np.eye(l_dim, dtype=np.float64) * self.r_noise
        self.gate_thresh = gate_thresh_val
    
    def force_relocalization(self):
        super().force_relocalization()
        l_dim = self.x.shape[0]
        if l_dim > 0:
            self.P = np.eye(l_dim, dtype=np.float64) * 1000.0

    def _compute_pixel_velocities(self, state_2n, v_cam, Z_est):
        """Berechnet s_dot aus Zustand x und Twist v_cam"""
        l_dim = state_2n.shape[0]
        n = l_dim // 2
        s_dot = np.zeros((l_dim, 1), dtype=np.float64)
        pts_pixel = state_2n.reshape(n, 2).T
        pts_norm = self.ibvs_math.pixel2normalized(pts_pixel)

        debug_prints_enabled = False

        if debug_prints_enabled:
            print("-"*30)
            print(f"Camera Twist (v_cam): {v_cam.flatten()}")
        
        for i in range(n):
            x_n = pts_norm[0, i]
            y_n = pts_norm[1, i]
            L_s = self.ibvs_math.get_interaction_matrix_point(x_n, y_n, Z_est)
            s_dot_norm = L_s @ v_cam
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
            if debug_prints_enabled:
                print(f"Feature {i}: pixel=({pts_pixel[0, i]:.1f}, {pts_pixel[1, i]:.1f}), "
                      f"norm=({x_n:.3f}, {y_n:.3f}), s_dot_norm=({s_dot_norm[0]:.3f}, {s_dot_norm[1]:.3f}), "
                      f"s_dot_pixel=({s_dot[i*2, 0]:.3f}, {s_dot[i*2+1, 0]:.3f})")
            
        return s_dot

    def predict(self, v_ee, Z_est, dt):
        if (not self.initialized) or self.x.shape[0] <= 0:
            return

        v_cam = self._transform_twist_ee_to_cam(v_ee)
        x_old = self.x.copy()
        s_dot = self._compute_pixel_velocities(x_old, v_cam, Z_est)
        l_dim = x_old.shape[0]
        F_k = np.eye(l_dim, dtype=np.float64)
        epsilon = 1e-4
        
        for i in range(l_dim):
            x_plus = x_old.copy()
            x_plus[i, 0] += epsilon
            s_dot_plus = self._compute_pixel_velocities(x_plus, v_cam, Z_est)
            diff = (s_dot_plus - s_dot) / epsilon
            F_k[:, i] += diff[:, 0] * dt

        self.x = x_old + s_dot * dt
        self.P = F_k @ self.P @ F_k.T + self.Q
        if self.update_geometry_from_state(self.x, log_on_fail=False):
            self.status = "PREDICT"
        else:
            self.status = "PREDICT (GEOMETRY HOLD)"

    def _build_observation_matrix(self, obs_slots):
        m = int(obs_slots.size)
        l_dim = self.x.shape[0]
        H_obs = np.zeros((2 * m, l_dim), dtype=np.float64)
        for j, slot in enumerate(obs_slots.tolist()):
            H_obs[2 * j, 2 * slot] = 1.0
            H_obs[2 * j + 1, 2 * slot + 1] = 1.0
        return H_obs

    def _get_active_state_vector(self):
        return self.x.copy()

    def update(self, current_pixels, desired_pixels, ref_ids=None, match_scores=None):
        z_k, obs_slots, prep_status = self._prepare_measurement(
            current_pixels,
            desired_pixels,
            ref_ids=ref_ids,
            match_scores=match_scores,
        )
        if z_k is None:
            self.status = "REJECT (NO MATCHES)"
            return

        self._set_state_dim(self.get_active_count())
        if self.x.shape[0] <= 0:
            self.status = "REJECT (NO ACTIVE)"
            return

        if not self.initialized:
            if z_k.shape[0] != self.x.shape[0]:
                self.status = "REJECT (INIT PARTIAL)"
                return
            self.x = z_k
            self.initialized = True
            self.update_geometry_from_state(self.x, log_on_fail=False)
            if prep_status == "RELOCALIZED":
                self.status = "RELOCALIZED"
            else:
                self.status = "INIT"
            return

        H_obs = self._build_observation_matrix(obs_slots)
        y = z_k - (H_obs @ self.x)
        r_obs = np.eye(z_k.shape[0], dtype=np.float64) * self.r_noise
        S = H_obs @ self.P @ H_obs.T + r_obs
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = float((y.T @ S_inv @ y)[0, 0]) / max(1.0, z_k.shape[0])
            if mahalanobis_dist > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                self.update_geometry_from_state(self.x, log_on_fail=False)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        K = self.P @ H_obs.T @ S_inv
        x_new = self.x + K @ y
        
        if self.update_geometry_from_state(x_new, log_on_fail=False):
            self.x = x_new
            self.P = (np.eye(self.x.shape[0], dtype=np.float64) - K @ H_obs) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
