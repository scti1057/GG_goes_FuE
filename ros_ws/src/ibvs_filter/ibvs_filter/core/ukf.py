# modelfilter_ibvs/core/filters/ukf.py
import numpy as np
import scipy.linalg
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class UnscentedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        self.L = 0
        self.x = np.zeros((0, 1), dtype=np.float64)
        self.P = np.zeros((0, 0), dtype=np.float64)
        self.Q = np.zeros((0, 0), dtype=np.float64)
        self.R = np.zeros((0, 0), dtype=np.float64)
        self.q_noise = 1.0
        self.r_noise = 50.0
        self.gate_thresh = 20.0
        
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lam = 0.0
        self.Wm = np.zeros((0,), dtype=np.float64)
        self.Wc = np.zeros((0,), dtype=np.float64)

    def _set_state_dim(self, active_count):
        l_new = int(2 * active_count)
        if l_new <= 0:
            self.L = 0
            self.x = np.zeros((0, 1), dtype=np.float64)
            self.P = np.zeros((0, 0), dtype=np.float64)
            self.Q = np.zeros((0, 0), dtype=np.float64)
            self.R = np.zeros((0, 0), dtype=np.float64)
            self.Wm = np.zeros((0,), dtype=np.float64)
            self.Wc = np.zeros((0,), dtype=np.float64)
            return
        if l_new == self.L:
            return
        self.L = l_new
        self.x = np.zeros((self.L, 1), dtype=np.float64)
        self.P = np.eye(self.L, dtype=np.float64) * 1000.0
        self.Q = np.eye(self.L, dtype=np.float64) * self.q_noise
        self.R = np.eye(self.L, dtype=np.float64) * self.r_noise
        self._recompute_ut_weights()

    def _recompute_ut_weights(self):
        if self.L <= 0:
            self.lam = 0.0
            self.Wm = np.zeros((0,), dtype=np.float64)
            self.Wc = np.zeros((0,), dtype=np.float64)
            return
        self.lam = (self.alpha**2) * (self.L + self.kappa) - self.L
        self.Wm = np.zeros(2 * self.L + 1, dtype=np.float64)
        self.Wc = np.zeros(2 * self.L + 1, dtype=np.float64)
        self.Wm[0] = self.lam / (self.L + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * self.L + 1):
            self.Wm[i] = 1.0 / (2 * (self.L + self.lam))
            self.Wc[i] = 1.0 / (2 * (self.L + self.lam))

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

    def _compute_pixel_velocities(self, state_2n, v_cam, Z_est):
        s_dot = np.zeros((self.L, 1))
        pts_norm = self.ibvs_math.pixel2normalized(state_2n.reshape(self.L // 2, 2).T)
        for i in range(self.L // 2):
            L_s = self.ibvs_math.get_interaction_matrix_point(pts_norm[0, i], pts_norm[1, i], Z_est)
            s_dot_norm = L_s @ v_cam
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
        return s_dot

    def _generate_sigma_points(self, x, P):
        """Generiert 2L+1 Sigma-Punkte mit robustem Cholesky-Schutz."""
        P_safe = (P + P.T) / 2.0 
        P_safe += np.eye(self.L) * 1e-8
        
        try:
            # Untere Dreiecksmatrix (lower=True)
            L_chol = scipy.linalg.cholesky(P_safe, lower=True)
        except scipy.linalg.LinAlgError:
            print("[UKF WARNING] Cholesky fehlgeschlagen, setze P zurück!")
            L_chol = np.eye(self.L) * 0.1 # Notfall-Fallback
            self.P = np.eye(self.L) * 10.0
            
        sigma_points = np.zeros((self.L, 2 * self.L + 1))
        sigma_points[:, 0] = x[:, 0]
        
        gamma = np.sqrt(self.L + self.lam)
        
        for i in range(self.L):
            sigma_points[:, i + 1]          = x[:, 0] + gamma * L_chol[:, i]
            sigma_points[:, self.L + i + 1] = x[:, 0] - gamma * L_chol[:, i]
            
        return sigma_points

    def predict(self, v_ee, Z_est, dt):
        if (not self.initialized) or self.L <= 0:
            return
        
        v_cam = self._transform_twist_ee_to_cam(v_ee)
        
        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_pred = np.zeros_like(sigmas)
        
        for i in range(2 * self.L + 1):
            x_i = sigmas[:, i].reshape(self.L, 1)
            s_dot_i = self._compute_pixel_velocities(x_i, v_cam, Z_est)
            sigmas_pred[:, i] = (x_i + s_dot_i * dt)[:, 0]
            
        x_pred = np.zeros((self.L, 1))
        for i in range(2 * self.L + 1):
            x_pred += self.Wm[i] * sigmas_pred[:, i].reshape(self.L, 1)
            
        P_pred = np.zeros((self.L, self.L))
        for i in range(2 * self.L + 1):
            y = sigmas_pred[:, i].reshape(self.L, 1) - x_pred
            P_pred += self.Wc[i] * (y @ y.T)
            
        self.x = x_pred
        self.P = P_pred + self.Q
        if self.update_geometry_from_state(self.x, log_on_fail=False):
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
        if self.L <= 0:
            self.status = "REJECT (NO ACTIVE)"
            return

        if not self.initialized:
            if z_k.shape[0] != self.L:
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
            nis = float((y.T @ S_inv @ y)[0, 0]) / max(1.0, z_k.shape[0])
            if nis > self.gate_thresh:
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
            self.P = (np.eye(self.L, dtype=np.float64) - K @ H_obs) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
