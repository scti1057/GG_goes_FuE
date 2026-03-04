# modelfilter_ibvs/core/filters/one_euro.py
import numpy as np
import cv2
from .base import BaseFilter

class OneEuroMath:
    """Mathematischer Kern des 1-Euro Filters (vektorisiert für Arrays)"""
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        # 1. Glättung der Ableitung (Geschwindigkeit)
        dx = (x - self.x_prev) / dt
        a_d = self.smoothing_factor(dt, self.d_cutoff)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 2. Dynamische Grenzfrequenz berechnen
        # Wir nutzen die L2-Norm (Magnitude) der Matrix-Änderung als skalare Geschwindigkeit
        speed = np.linalg.norm(dx_hat) 
        cutoff = self.min_cutoff + self.beta * speed

        # 3. Position glätten
        a = self.smoothing_factor(dt, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Zustand speichern
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


class OneEuroFilter(BaseFilter):
    """
    Implementiert den 1-Euro Filter für IBVS.
    Schätzt eine Homographie aus den XFeat-Matches und filtert deren Parameter.
    """
    def __init__(self, K):
        super().__init__(K)
        # Wir filtern ein 9-Elemente-Array (die Parameter der 3x3 Homographie)
        self.euro_core = OneEuroMath(min_cutoff=0.1, beta=0.5) 
        self.H_filtered = np.eye(3)
        self.last_dt = 0.05 # Fallback

    def reset(self):
        super().reset()
        self.euro_core = OneEuroMath(min_cutoff=0.1, beta=0.5)
        self.H_filtered = np.eye(3)

    def predict(self, v_ee, Z_est, dt):
        # Der 1-Euro Filter ist reaktiv und hat kein physikalisches Modell.
        # Wir merken uns nur das dt für den Update-Schritt.
        self.last_dt = dt
        self.status = "PREDICT" # Wir setzen Status, falls kein Update folgt

    def update(self, current_pixels, desired_pixels):
        # Wenn wir geblendet wurden oder xfeat versagt hat
        if current_pixels is None or current_pixels.shape[1] < 4:
            self.status = "MISSING"
            return

        # 1. Rohe Homographie mit RANSAC schätzen
        # findHomography erwartet Shape (N, 1, 2) oder (N, 2). Wir haben (2, N).
        src_pts = desired_pixels.T.astype(np.float32)
        dst_pts = current_pixels.T.astype(np.float32)

        H_raw, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H_raw is not None:
            # 2. Homographie als 1D-Array (9 Elemente) glätten
            h_flat_raw = H_raw.flatten()
            
            h_flat_filtered = self.euro_core.filter(h_flat_raw, self.last_dt)
            
            # 3. Zurück in eine 3x3 Matrix wandeln und normieren (H_33 sollte nahe 1 sein)
            H_filt = h_flat_filtered.reshape(3, 3)
            if H_filt[2, 2] != 0:
                self.H_filtered = H_filt / H_filt[2, 2]
            else:
                self.H_filtered = H_filt
                
            self.initialized = True
            self.status = "UPDATE"
        else:
            self.status = "REJECT (RANSAC FAIL)"

    def get_projected_points(self, desired_features):
        if not self.initialized:
            return desired_features # Fallback, wenn noch nicht initialisiert
            
        # desired_features ist (2, M). Mache homogen: (3, M)
        M = desired_features.shape[1]
        hom_pts = np.vstack((desired_features, np.ones((1, M))))
        
        # Multipliziere mit gefilterter Homographie
        proj_hom = self.H_filtered @ hom_pts
        
        # Zurück nach 2D (durch Z-Koordinate teilen)
        u = proj_hom[0, :] / proj_hom[2, :]
        v = proj_hom[1, :] / proj_hom[2, :]
        
        return np.vstack((u, v))