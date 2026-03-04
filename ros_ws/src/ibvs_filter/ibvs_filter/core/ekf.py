# modelfilter_ibvs/core/filters/ekf.py
import numpy as np
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class ExtendedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        
        # Zustand x ist nun 8D! (Nur noch die Positionen, KEINE Geschwindigkeiten mehr)
        # Die Geschwindigkeit wird ja durch v_ee deterministisch berechnet!
        self.x = np.zeros((8, 1))
        self.P = np.eye(8) * 1000.0

        # H ist nun einfach die Einheitsmatrix (da Messung und Zustand gleich sind)
        self.H = np.eye(8) # Mess-Jacobi ist I

        # Tuning Parameter
        self.Q = np.eye(8) * 1.0  # Q kann jetzt viel kleiner sein als im SKF!
        self.R = np.eye(8) * 50.0
        self.gate_thresh = 20.0

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.Q = np.eye(8) * q_val
        self.R = np.eye(8) * r_val
        self.gate_thresh = gate_thresh_val
    
    def force_relocalization(self):
        super().force_relocalization()
        # Setze die Unsicherheit wieder hoch, da wir quasi von vorne anfangen
        self.P = np.eye(8) * 1000.0

    def _transform_twist_ee_to_cam(self, v_ee):
        """Transformiert den Twist vom Endeffektor- ins Kamerakoordinatensystem"""
        # In robot_sim.py: T_ee_cam = SE3(0, 0, 0) * SE3.Rz(np.pi)
        # Drehung um 180 Grad um Z bedeutet: x' = -x, y' = -y, z' = z
        v_cam = np.zeros(6)
        v_cam[0] = -v_ee[0] # vx
        v_cam[1] = -v_ee[1] # vy
        v_cam[2] =  v_ee[2] # vz
        v_cam[3] = -v_ee[3] # wx
        v_cam[4] = -v_ee[4] # wy
        v_cam[5] =  v_ee[5] # wz
        return v_cam

    def _compute_pixel_velocities(self, state_8d, v_cam, Z_est):
        """Berechnet s_dot aus Zustand x und Twist v_cam"""
        s_dot = np.zeros((8, 1))
        # Wandle 8D Vektor in 2x4 Matrix für die IBVS Math
        pts_pixel = state_8d.reshape(4, 2).T
        pts_norm = self.ibvs_math.pixel2normalized(pts_pixel)
        
        for i in range(4):
            x_n = pts_norm[0, i]
            y_n = pts_norm[1, i]
            L_s = self.ibvs_math.get_interaction_matrix_point(x_n, y_n, Z_est)
            # L_s ist 2x6, v_cam ist 6x1 -> s_dot_i ist 2x1 (in normalisierten Koordinaten)
            s_dot_norm = L_s @ v_cam
            # Zurück in Pixelgeschwindigkeit umrechnen (mit Brennweiten f_x, f_y)
            s_dot[i*2, 0] = s_dot_norm[0] * self.K[0, 0]
            s_dot[i*2+1, 0] = s_dot_norm[1] * self.K[1, 1]
            
        return s_dot

    def predict(self, v_ee, Z_est, dt):
        if not self.initialized:
            return

        # 1. Kinematik umrechnen
        v_cam = self._transform_twist_ee_to_cam(v_ee)
        
        # WICHTIG: Den alten Zustand einfrieren, bevor wir irgendetwas berechnen!
        x_old = self.x.copy()
        
        # 2. Pixelgeschwindigkeit am ALTEN Zustand berechnen
        s_dot = self._compute_pixel_velocities(x_old, v_cam, Z_est)
        
        # 3. Numerische Jacobi-Matrix F_k berechnen (ebenfalls am ALTEN Zustand!)
        F_k = np.eye(8)
        epsilon = 1e-4
        
        for i in range(8):
            x_plus = x_old.copy() # Richtig: Ausgehend vom alten Zustand!
            x_plus[i, 0] += epsilon
            
            s_dot_plus = self._compute_pixel_velocities(x_plus, v_cam, Z_est)
            
            # Numerische Ableitung: Änderung der Geschwindigkeit bezogen auf epsilon
            diff = (s_dot_plus - s_dot) / epsilon
            F_k[:, i] += diff[:, 0] * dt

        # 4. Zustand aktualisieren (Prädiktion ausführen)
        self.x = x_old + s_dot * dt

        # 5. Kovarianz-Update
        self.P = F_k @ self.P @ F_k.T + self.Q
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels):
        # 1. Messung über Base-Class generieren
        z_k, H_raw, self.status = self._get_raw_measurement(current_pixels, desired_pixels)
        
        if z_k is None:
            return

        # 2. Initialisierung / Relokalisation abfangen
        if not self.initialized:
            self.x = z_k
            self.initialized = True
            self._check_geometry_and_update_H(self.x)
            self.status = "INIT"
            return

        # 3. Kalman Update
        y = z_k - self.x
        S = self.P + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = y.T @ S_inv @ y
            if mahalanobis_dist > self.gate_thresh:
                self.status = "REJECT (OUTLIER)"
                self._check_geometry_and_update_H(self.x)
                return
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"
            return

        K = self.P @ S_inv
        x_new = self.x + K @ y
        
        # 4. Geometrie checken, bevor wir den Zustand endgültig übernehmen
        if self._check_geometry_and_update_H(x_new):
            self.x = x_new
            self.P = (np.eye(8) - K) @ self.P
            self.status = "UPDATE"
        else:
            self.status = "REJECT (GEOMETRY)"