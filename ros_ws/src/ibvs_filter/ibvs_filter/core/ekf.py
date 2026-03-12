# modelfilter_ibvs/core/filters/ekf.py
import numpy as np
from .base import BaseFilter
from .ibvs_math import IBVSMath # Für die Interaktionsmatrix

class ExtendedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.ibvs_math = IBVSMath(K)
        
        # KONFIGURATION: Anzahl der Tracking-Slots
        self.N_SLOTS = 12 
        self.state_dim = self.N_SLOTS * 2
        
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1000.0

        # Map: Ref_ID -> Slot_Index (0 bis N_SLOTS-1)
        self.track_map = {}

        # Tuning Parameter
        self.Q = np.eye(self.state_dim) * 1.0 
        self.R = np.eye(self.state_dim) * 50.0
        # Wir speichern R_val als Skalar, da wir R im Update dynamisch bauen
        self.r_val = 50.0 
        self.gate_thresh = 30.0 # Etwas höher, da Features springen können

    def set_Q_R_gate(self, q_val, r_val, gate_thresh_val):
        self.Q = np.eye(self.state_dim) * q_val
        self.r_val = r_val
        self.gate_thresh = gate_thresh_val
    
    def force_relocalization(self):
        super().force_relocalization()
        self.P = np.eye(self.state_dim) * 1000.0
        self.track_map = {} # Alle Slots freigeben
        self.x = np.zeros((self.state_dim, 1))

    def _compute_pixel_velocities(self, state_8d, v_cam, Z_est):
        """
        Berechnet s_dot für den aktuellen State Vektor.
        state_8d ist hier eigentlich state_Nd (2*N).
        """
        s_dot = np.zeros_like(state_8d)
        
        # Reshape für IBVS Math: (N, 2) -> (2, N)
        pts_pixel = state_8d.reshape(self.N_SLOTS, 2).T
        pts_norm = self.ibvs_math.pixel2normalized(pts_pixel)
        
        for i in range(self.N_SLOTS):
            # Nur berechnen, wenn der Slot belegt ist (einfacher Check: nicht 0)
            if state_8d[i*2, 0] == 0 and state_8d[i*2+1, 0] == 0:
                continue

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
        F_k = np.eye(self.state_dim)
        epsilon = 1e-4
        
        for i in range(self.state_dim):
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
        
        # Proxy Update nur für Visualisierung (Bounding Box der Punktwolke)
        self._update_proxy_visuals()
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels, ref_ids=None):
        if ref_ids is None or len(ref_ids) == 0:
            return

        # 1. Slot Management: Neue Features initialisieren, falls Slots frei sind
        # Wir machen das sehr simpel: Wenn wir eine ID noch nicht kennen und Platz haben, rein damit.
        
        # Verfügbare freie Slots finden
        used_slots = set(self.track_map.values())
        free_slots = [s for s in range(self.N_SLOTS) if s not in used_slots]
        
        for i, r_id in enumerate(ref_ids):
            if r_id not in self.track_map:
                if len(free_slots) > 0:
                    slot = free_slots.pop(0)
                    self.track_map[r_id] = slot
                    # Initialisierung des Slots mit Messwert
                    self.x[slot*2, 0] = current_pixels[0, i]
                    self.x[slot*2+1, 0] = current_pixels[1, i]
                    # Unsicherheit für diesen neuen Punkt hochsetzen? 
                    # Optional, aber P ist global gekoppelt, lassen wir es erst mal so.

        # 2. Messvektor z und Matrix H dynamisch bauen
        # Wir updaten NUR die Slots, für die wir aktuell eine Messung haben.
        rows_z = []
        rows_H_indices = []
        obs_count = 0
        
        for i, r_id in enumerate(ref_ids):
            if r_id in self.track_map:
                slot = self.track_map[r_id]
                
                # Messung u
                rows_z.append(current_pixels[0, i])
                rows_H_indices.append(slot * 2)
                
                # Messung v
                rows_z.append(current_pixels[1, i])
                rows_H_indices.append(slot * 2 + 1)
                
                obs_count += 1

        if obs_count == 0:
            return

        if not self.initialized:
            self.initialized = True
            self.status = "INIT"

        # 3. EKF Update Schritte
        z = np.array(rows_z).reshape(-1, 1)
        
        # H ist sparse (selektiert Zeilen aus der Identitätsmatrix)
        H_sparse = np.zeros((len(rows_z), self.state_dim))
        for k, state_idx in enumerate(rows_H_indices):
            H_sparse[k, state_idx] = 1.0

        # R Matrix dynamisch (Skalar auf Diagonale)
        R_dynamic = np.eye(len(rows_z)) * self.r_val

        # Prädizierte Messung
        y = z - (H_sparse @ self.x)
        
        S = H_sparse @ self.P @ H_sparse.T + R_dynamic
        
        try:
            S_inv = np.linalg.inv(S)
            
            # Gating Skalierung: Threshold wächst mit Anzahl der Messungen
            if (y.T @ S_inv @ y) > (self.gate_thresh * obs_count / 2.0): 
                self.status = "REJECT (OUTLIER)"
                return

            K = self.P @ H_sparse.T @ S_inv
            self.x = self.x + K @ y
            self.P = (np.eye(self.state_dim) - K @ H_sparse) @ self.P
            
            self._update_proxy_visuals()
            self.status = f"UPDATE ({obs_count}/{self.N_SLOTS})"
            
        except np.linalg.LinAlgError:
            self.status = "REJECT (SINGULAR)"

    def _update_proxy_visuals(self):
        """Berechnet Bounding Box um alle aktiven Punkte für Visualisierung"""
        pts = self.x.reshape(self.N_SLOTS, 2)
        valid = pts[np.any(pts != 0, axis=1)] # Filter (0,0)
        if len(valid) < 2: return
        
        u_min, v_min = np.min(valid, axis=0)
        u_max, v_max = np.max(valid, axis=0)
        
        # Wir setzen proxy_ref direkt, damit BaseFilter.get_proxy_corners was liefert
        self.proxy_ref = np.array([
            [u_min, v_min], [u_max, v_min],
            [u_max, v_max], [u_min, v_max]
        ], dtype=np.float32)
        # Hack: Da wir keine Homographie mehr haben, nutzen wir BaseFilter Logik nicht mehr
        # Wir überschreiben get_proxy_corners einfach in base oder hier.
        # Einfachheitshalber nutzen wir proxy_ref als 'est', da dies die gefilterten Punkte sind.
    
    def get_proxy_corners(self):
        # Wir liefern Dummy für Ref (orange) und unsere Bounding Box für Est (cyan)
        # Da 'self.proxy_ref' oben eigentlich die gefilterte Box ist:
        if self.proxy_ref is None: return None, None
        return np.zeros((4,2)), self.proxy_ref
