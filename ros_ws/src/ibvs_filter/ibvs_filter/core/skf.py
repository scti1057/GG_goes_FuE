# modelfilter_ibvs/core/filters/skf.py
import numpy as np
from .base import BaseFilter

class StandardKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        
        # Zustand x (16x1): [u1, v1, u2, v2, u3, v3, u4, v4, u_dot1, v_dot1, ...]
        self.x = np.zeros((16, 1))
        self.P = np.eye(16) * 1000.0  # Hohe initiale Unsicherheit
        
        self.F = np.eye(16)
        # H projiziert 16D Zustand auf 8D Messung (nur Positionen)
        self.H = np.zeros((8, 16))
        for i in range(8):
            self.H[i, i] = 1.0
            
        self.Q = np.eye(16) * 100.0
        self.R = np.eye(8) * 50.0
        self.gate_thresh = 20.0

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        """Wird von der GUI (SpinBoxes) aufgerufen, um Tuning live anzupassen.
        Tuning: Geschwindigkeiten bekommen q_val, Positionen etwas weniger"""
        # Prozessrauschen hauptsächlich auf die Geschwindigkeiten (Indizes 8 bis 15)
        # und leicht auf die Positionen (0 bis 7)
        self.Q = np.eye(16) * (q_val * 0.1) 
        for i in range(8, 16):
            self.Q[i, i] = q_val
            
        self.R = np.eye(8) * r_val
        self.gate_thresh = gate_thresh_val

    def force_relocalization(self):
        super().force_relocalization()
        self.P = np.eye(16) * 1000.0 # Unsicherheit komplett zurücksetzen

    def predict(self, v_ee, Z_est, dt):
        if not self.initialized:
            return

        # Das 3D Twist-Kommando (v_ee) ignorieren wir im SKF.
        # Wir updaten nur die F-Matrix für das Constant Velocity Modell mit dt.
        for i in range(8):
            self.F[i, i + 8] = dt
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.update_geometry_from_state(self.x[0:8], log_on_fail=False):
            self.status = "PREDICT"
        else:
            self.status = "PREDICT (GEOMETRY HOLD)"

    def update(self, current_pixels, desired_pixels):
        # 1. Messung über Base-Class generieren (dynamische Bounding-Box greift hier!)
        z_k, H_raw, self.status = self._get_raw_measurement(current_pixels, desired_pixels)

        if z_k is None:
            return

        # 2. Initialisierung / Relokalisation
        if not self.initialized:
            self.x[0:8] = z_k
            self.x[8:16] = 0 # Geschwindigkeiten auf 0 initialisieren
            self.initialized = True
            self._check_geometry_and_update_H(self.x[0:8])
            self.status = "INIT"
            return

        # 3. Kalman Update
        y = z_k - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        
        # Gating
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = y.T @ S_inv @ y
            
            if mahalanobis_dist > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                # H_filtered trotzdem aus der (rein prädizierten) Position berechnen
                self._check_geometry_and_update_H(self.x[0:8])
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        # Gain und Update
        K = self.P @ self.H.T @ S_inv
        x_new = self.x + K @ y
        
        # 4. Geometrie checken (Wir übergeben nur die Positionen x_new[0:8])
        if self._check_geometry_and_update_H(x_new[0:8]):
            self.x = x_new
            self.P = (np.eye(16) - K @ self.H) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"
