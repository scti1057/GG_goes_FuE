# modelfilter_ibvs/core/filters/eskf.py
import numpy as np
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class ErrorStateKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        self.L = 0
        self.x_nom = np.zeros((0, 1), dtype=np.float64)
        self.dx = np.zeros((0, 1), dtype=np.float64)
        self.P = np.zeros((0, 0), dtype=np.float64)
        self.q_noise = 1.0
        self.r_noise = 50.0
        self.Q = np.zeros((0, 0), dtype=np.float64)
        self.R = np.zeros((0, 0), dtype=np.float64)
        self.gate_thresh = 20.0

    def _set_state_dim(self, active_count):
        l_new = int(2 * active_count)
        if l_new <= 0:
            self.L = 0
            self.x_nom = np.zeros((0, 1), dtype=np.float64)
            self.dx = np.zeros((0, 1), dtype=np.float64)
            self.P = np.zeros((0, 0), dtype=np.float64)
            self.Q = np.zeros((0, 0), dtype=np.float64)
            self.R = np.zeros((0, 0), dtype=np.float64)
            return
        if l_new == self.L:
            return
        self.L = l_new
        self.x_nom = np.zeros((self.L, 1), dtype=np.float64)
        self.dx = np.zeros((self.L, 1), dtype=np.float64)
        self.P = np.eye(self.L, dtype=np.float64) * 1000.0
        self.Q = np.eye(self.L, dtype=np.float64) * self.q_noise
        self.R = np.eye(self.L, dtype=np.float64) * self.r_noise

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.q_noise = float(q_val)
        self.r_noise = float(r_val)
        self.Q = np.eye(self.L, dtype=np.float64) * self.q_noise
        self.R = np.eye(self.L, dtype=np.float64) * self.r_noise
        self.gate_thresh = gate_thresh_val

    def force_relocalization(self):
        super().force_relocalization()
        if self.L > 0:
            self.P = np.eye(self.L, dtype=np.float64) * 1000.0
            self.dx = np.zeros((self.L, 1), dtype=np.float64)

    def _compute_pixel_velocities(self, state_2n, v_cam, Z_est):
        s_dot = np.zeros((self.L, 1))
        pts_norm = self.ibvs_math.pixel2normalized(state_2n.reshape(self.L // 2, 2).T)
        for i in range(self.L // 2):
            L_s = self.ibvs_math.get_interaction_matrix_point(pts_norm[0, i], pts_norm[1, i], Z_est)
            s_dot_norm = L_s @ v_cam
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
        return s_dot

    def predict(self, v_ee, Z_est, dt):
        if (not self.initialized) or self.L <= 0:
            return
        
        v_cam = self._transform_twist_ee_to_cam(v_ee)
        
        x_nom_old = self.x_nom.copy()
        s_dot = self._compute_pixel_velocities(x_nom_old, v_cam, Z_est)
        
        self.x_nom = x_nom_old + s_dot * dt
        
        F_dx = np.eye(self.L, dtype=np.float64)
        epsilon = 1e-4
        for i in range(self.L):
            x_plus = x_nom_old.copy()
            x_plus[i, 0] += epsilon
            s_dot_plus = self._compute_pixel_velocities(x_plus, v_cam, Z_est)
            F_dx[:, i] += ((s_dot_plus - s_dot) / epsilon)[:, 0] * dt

        self.P = F_dx @ self.P @ F_dx.T + self.Q
        if self.update_geometry_from_state(self.x_nom, log_on_fail=False):
            self.status = "PREDICT"
        else:
            self.status = "PREDICT (GEOMETRY HOLD)"

    def _build_observation_matrix(self, obs_slots):
        m = int(obs_slots.size)
        H_obs = np.zeros((2 * m, self.L), dtype=np.float64)
        for j, slot in enumerate(obs_slots.tolist()):
            H_obs[2 * j, 2 * slot] = 1.0
            H_obs[2 * j + 1, 2 * slot + 1] = 1.0
        return H_obs

    def _get_active_state_vector(self):
        return self.x_nom.copy()

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
        if self.L <= 0:
            self.status = "REJECT (NO ACTIVE)"
            return

        if not self.initialized:
            if z_k.shape[0] != self.L:
                self.status = "REJECT (INIT PARTIAL)"
                return
            self.x_nom = z_k
            self.dx = np.zeros((self.L, 1), dtype=np.float64)
            self.initialized = True
            self.update_geometry_from_state(self.x_nom, log_on_fail=False)
            if prep_status == "RELOCALIZED":
                self.status = "RELOCALIZED"
            else:
                self.status = "INIT"
            return

        H_obs = self._build_observation_matrix(obs_slots)
        y = z_k - (H_obs @ self.x_nom)
        r_obs = np.eye(z_k.shape[0], dtype=np.float64) * self.r_noise
        S = H_obs @ self.P @ H_obs.T + r_obs
        
        try:
            S_inv = np.linalg.inv(S)
            nis = float((y.T @ S_inv @ y)[0, 0]) / max(1.0, z_k.shape[0])
            if nis > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                self.update_geometry_from_state(self.x_nom, log_on_fail=False)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        K = self.P @ H_obs.T @ S_inv
        self.dx = K @ y
        self.P = (np.eye(self.L, dtype=np.float64) - K @ H_obs) @ self.P
        x_new = self.x_nom + self.dx
        
        if self.update_geometry_from_state(x_new, log_on_fail=False):
            self.x_nom = x_new
            self.dx = np.zeros((self.L, 1), dtype=np.float64)
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
