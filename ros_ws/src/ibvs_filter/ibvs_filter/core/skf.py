# modelfilter_ibvs/core/filters/skf.py
import numpy as np
from .base import BaseFilter

class StandardKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.keypoint_count = 0
        self.pos_dim = 0
        self.state_dim = 0
        self.x = np.zeros((0, 1), dtype=np.float64)
        self.P = np.zeros((0, 0), dtype=np.float64)
        self.F = np.zeros((0, 0), dtype=np.float64)
        self.q_noise = 100.0
        self.r_noise = 50.0
        self.Q = np.zeros((0, 0), dtype=np.float64)
        self.R = np.zeros((0, 0), dtype=np.float64)
        self.gate_thresh = 20.0

    def _set_state_dim(self, active_count):
        n = int(active_count)
        if n <= 0:
            self.keypoint_count = 0
            self.pos_dim = 0
            self.state_dim = 0
            self.x = np.zeros((0, 1), dtype=np.float64)
            self.P = np.zeros((0, 0), dtype=np.float64)
            self.F = np.zeros((0, 0), dtype=np.float64)
            self.Q = np.zeros((0, 0), dtype=np.float64)
            self.R = np.zeros((0, 0), dtype=np.float64)
            return
        if n == self.keypoint_count:
            return
        self.keypoint_count = n
        self.pos_dim = 2 * n
        self.state_dim = 4 * n
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1000.0
        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.Q = np.eye(self.state_dim, dtype=np.float64) * (self.q_noise * 0.1)
        if self.state_dim > self.pos_dim:
            self.Q[self.pos_dim:, self.pos_dim:] = (
                np.eye(self.state_dim - self.pos_dim, dtype=np.float64) * self.q_noise
            )
        self.R = np.eye(self.pos_dim, dtype=np.float64) * self.r_noise

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.q_noise = float(q_val)
        self.r_noise = float(r_val)
        self.Q = np.eye(self.state_dim, dtype=np.float64) * (self.q_noise * 0.1)
        if self.state_dim > self.pos_dim:
            self.Q[self.pos_dim:, self.pos_dim:] = (
                np.eye(self.state_dim - self.pos_dim, dtype=np.float64) * self.q_noise
            )
        self.R = np.eye(self.pos_dim, dtype=np.float64) * self.r_noise
        self.gate_thresh = gate_thresh_val

    def force_relocalization(self):
        super().force_relocalization()
        if self.state_dim > 0:
            self.P = np.eye(self.state_dim, dtype=np.float64) * 1000.0

    def predict(self, v_ee, Z_est, dt):
        if (not self.initialized) or self.state_dim <= 0:
            return

        self.F = np.eye(self.state_dim, dtype=np.float64)
        for i in range(self.pos_dim):
            self.F[i, i + self.pos_dim] = dt
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.update_geometry_from_state(self.x[0:self.pos_dim], log_on_fail=False):
            self.status = "PREDICT"
        else:
            self.status = "PREDICT (GEOMETRY HOLD)"

    def _build_observation_matrix(self, obs_slots):
        m = int(obs_slots.size)
        H_obs = np.zeros((2 * m, self.state_dim), dtype=np.float64)
        for j, slot in enumerate(obs_slots.tolist()):
            H_obs[2 * j, 2 * slot] = 1.0
            H_obs[2 * j + 1, 2 * slot + 1] = 1.0
        return H_obs

    def _get_active_state_vector(self):
        return self.x[0:self.pos_dim].copy()

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
        if self.state_dim <= 0:
            self.status = "REJECT (NO ACTIVE)"
            return

        if not self.initialized:
            if z_k.shape[0] != self.pos_dim:
                self.status = "REJECT (INIT PARTIAL)"
                return
            self.x[0:self.pos_dim] = z_k
            self.x[self.pos_dim:self.state_dim] = 0.0
            self.initialized = True
            self.update_geometry_from_state(self.x[0:self.pos_dim], log_on_fail=False)
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
                self.update_geometry_from_state(self.x[0:self.pos_dim], log_on_fail=False)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        K = self.P @ H_obs.T @ S_inv
        x_new = self.x + K @ y
        
        if self.update_geometry_from_state(x_new[0:self.pos_dim], log_on_fail=False):
            self.x = x_new
            self.P = (np.eye(self.state_dim, dtype=np.float64) - K @ H_obs) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
