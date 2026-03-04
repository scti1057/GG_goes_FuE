# modelfilter_ibvs/core/filters/ukf.py
import numpy as np
import scipy.linalg
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class UnscentedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        
        self.L = 8 # Dimension des Zustands (4 Punkte x 2 Koordinaten)
        self.x = np.zeros((self.L, 1))
        self.P = np.eye(self.L) * 1000.0
        
        # Tuning für den UKF
        self.Q = np.eye(self.L) * 1.0
        self.R = np.eye(self.L) * 50.0
        self.gate_thresh = 20.0
        
        # UT Parameter nach Wan & van der Merwe
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lam = (self.alpha**2) * (self.L + self.kappa) - self.L
        
        # Gewichte W_m (Mittelwert) und W_c (Kovarianz) vorberechnen
        self.Wm = np.zeros(2 * self.L + 1)
        self.Wc = np.zeros(2 * self.L + 1)
        
        self.Wm[0] = self.lam / (self.L + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.L + 1):
            self.Wm[i] = 1.0 / (2 * (self.L + self.lam))
            self.Wc[i] = 1.0 / (2 * (self.L + self.lam))

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.Q = np.eye(self.L) * q_val
        self.R = np.eye(self.L) * r_val
        self.gate_thresh = gate_thresh_val

    def force_relocalization(self):
        super().force_relocalization()
        self.P = np.eye(self.L) * 1000.0 

    def _transform_twist_ee_to_cam(self, v_ee):
        v_cam = np.zeros(6)
        v_cam[0], v_cam[1], v_cam[2] = -v_ee[0], -v_ee[1], v_ee[2]
        v_cam[3], v_cam[4], v_cam[5] = -v_ee[3], -v_ee[4], v_ee[5]
        return v_cam

    def _compute_pixel_velocities(self, state_8d, v_cam, Z_est):
        s_dot = np.zeros((self.L, 1))
        pts_norm = self.ibvs_math.pixel2normalized(state_8d.reshape(4, 2).T)
        for i in range(4):
            L_s = self.ibvs_math.get_interaction_matrix_point(pts_norm[0, i], pts_norm[1, i], Z_est)
            s_dot_norm = L_s @ v_cam
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
        return s_dot

    def _generate_sigma_points(self, x, P):
        """Generiert 2L+1 Sigma-Punkte mit robustem Cholesky-Schutz."""
        # 1. Numerische Stabilisierung (Schutz vor Cholesky-Crash)
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
        if not self.initialized: return
        
        v_cam = self._transform_twist_ee_to_cam(v_ee)
        
        # 1. Sigma-Punkte generieren
        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_pred = np.zeros_like(sigmas)
        
        # 2. Sigma-Punkte durch die nichtlineare Kinematik jagen (f(x, u))
        for i in range(2 * self.L + 1):
            x_i = sigmas[:, i].reshape(self.L, 1)
            # Prädiktion: x_neu = x_alt + geschwindigkeit * dt
            s_dot_i = self._compute_pixel_velocities(x_i, v_cam, Z_est)
            sigmas_pred[:, i] = (x_i + s_dot_i * dt)[:, 0]
            
        # 3. Mittelwert und Kovarianz rekonstruieren
        x_pred = np.zeros((self.L, 1))
        for i in range(2 * self.L + 1):
            x_pred += self.Wm[i] * sigmas_pred[:, i].reshape(self.L, 1)
            
        P_pred = np.zeros((self.L, self.L))
        for i in range(2 * self.L + 1):
            y = sigmas_pred[:, i].reshape(self.L, 1) - x_pred
            P_pred += self.Wc[i] * (y @ y.T)
            
        # Prozessrauschen addieren
        self.x = x_pred
        self.P = P_pred + self.Q
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels):
        # Das Update ist identisch zum EKF, da unser Messmodell H=I ist (Semi-Nichtlinear)
        z_k, H_raw, self.status = self._get_raw_measurement(current_pixels, desired_pixels)
        
        if z_k is None:
            return

        if not self.initialized:
            self.x = z_k
            self.initialized = True
            self._check_geometry_and_update_H(self.x)
            self.status = "INIT"
            return

        y = z_k - self.x
        S = self.P + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            if (y.T @ S_inv @ y) > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                self._check_geometry_and_update_H(self.x)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        K = self.P @ S_inv
        x_new = self.x + K @ y
        
        if self._check_geometry_and_update_H(x_new):
            self.x = x_new
            self.P = (np.eye(self.L) - K) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"