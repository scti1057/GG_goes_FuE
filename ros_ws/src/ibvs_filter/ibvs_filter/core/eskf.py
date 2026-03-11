# modelfilter_ibvs/core/filters/eskf.py
import numpy as np
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class ErrorStateKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        
        self.L = 8
        
        # 1. Die Zustandsaufteilung
        self.x_nom = np.zeros((self.L, 1)) # Nominalzustand (Die echte physikalische Schätzung)
        self.dx = np.zeros((self.L, 1))    # Fehlerzustand (Wird vom Kalman-Filter geschätzt)
        
        self.P = np.eye(self.L) * 1000.0   # Kovarianz des FEHLERS!
        
        # Tuning
        self.Q = np.eye(self.L) * 1.0
        self.R = np.eye(self.L) * 50.0
        self.gate_thresh = 20.0

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.Q = np.eye(self.L) * q_val
        self.R = np.eye(self.L) * r_val
        self.gate_thresh = gate_thresh_val

    def force_relocalization(self):
        super().force_relocalization()
        self.P = np.eye(self.L) * 1000.0
        self.dx = np.zeros((self.L, 1)) # Fehler ebenfalls zurücksetzen

    def _compute_pixel_velocities(self, state_8d, v_cam, Z_est):
        s_dot = np.zeros((self.L, 1))
        pts_norm = self.ibvs_math.pixel2normalized(state_8d.reshape(4, 2).T)
        for i in range(4):
            L_s = self.ibvs_math.get_interaction_matrix_point(pts_norm[0, i], pts_norm[1, i], Z_est)
            s_dot_norm = L_s @ v_cam
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
        return s_dot

    def predict(self, v_ee, Z_est, dt):
        if not self.initialized: return
        
        v_cam = self._transform_twist_ee_to_cam(v_ee)
        
        # --- PHASE 1: Nominale Prädiktion (Pure Kinematik) ---
        x_nom_old = self.x_nom.copy()
        s_dot = self._compute_pixel_velocities(x_nom_old, v_cam, Z_est)
        
        self.x_nom = x_nom_old + s_dot * dt
        # Hinweis: self.dx bleibt hier unberührt (ist 0)!
        
        # --- PHASE 2a: Fehler-Kovarianz prädizieren ---
        # Die Fehler-Jacobi F_dx ist identisch zur System-Jacobi F_x,
        # da unser Vektorraum euklidisch ist (R^8).
        F_dx = np.eye(self.L)
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

    def update(self, current_pixels, desired_pixels):
        z_k, H_raw, self.status = self._get_raw_measurement(current_pixels, desired_pixels)
        
        if z_k is None:
            return

        if not self.initialized:
            # Harter Reset bei Initialisierung
            self.x_nom = z_k
            self.dx = np.zeros((self.L, 1))
            self.initialized = True
            self._check_geometry_and_update_H(self.x_nom)
            self.status = "INIT"
            return

        # --- PHASE 2b: Fehler-Filterung (Update) ---
        # Das Residuum ist die Differenz zwischen Messung und Nominalzustand.
        # Streng genommen: y = z_k - (x_nom + dx). Da dx vor dem Update immer 0 ist, reicht z_k - x_nom.
        y = z_k - self.x_nom
        
        S = self.P + self.R # Da H = I, ist H*P*H^T = P
        
        try:
            S_inv = np.linalg.inv(S)
            if (y.T @ S_inv @ y) > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                self._check_geometry_and_update_H(self.x_nom)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        # Kalman-Gain berechnen
        K = self.P @ S_inv
        
        # Fehler schätzen!
        self.dx = K @ y
        
        # Kovarianz aktualisieren
        self.P = (np.eye(self.L) - K) @ self.P
        
        # --- PHASE 3: Injektion und Reset ---
        # Fehler in den Nominalzustand injizieren
        x_new = self.x_nom + self.dx
        
        # Geometrie checken (mit dem injizierten neuen Zustand)
        if self._check_geometry_and_update_H(x_new):
            self.x_nom = x_new
            
            # WICHTIGSTER SCHRITT IM ESKF: Fehler zurücksetzen!
            self.dx = np.zeros((self.L, 1))
            
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
