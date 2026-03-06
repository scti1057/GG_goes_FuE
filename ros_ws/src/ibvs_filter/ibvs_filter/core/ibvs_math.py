import numpy as np

class IBVSMath:
    """
    Stellt die rein mathematischen Funktionen für die Bild-Kinematik (Interaktionsmatrix)
    bereit, ohne Abhängigkeiten zu externen Regler-Klassen.
    """
    def __init__(self, K):
        self.K = K
        # Wir invertieren K einmalig, um später Rechenzeit zu sparen
        self.K_inv = np.linalg.inv(K)

    def pixel2normalized(self, pixels):
        """
        Wandelt Pixelkoordinaten [u, v] in normalisierte Bildkoordinaten [x, y] um.
        :param pixels: 2xN Numpy Array
        :return: 2xN Numpy Array
        """
        N = pixels.shape[1]
        hom_pixels = np.vstack((pixels, np.ones((1, N))))
        norm_pixels = self.K_inv @ hom_pixels
        return norm_pixels[0:2, :]

    def get_interaction_matrix_point(self, x, y, Z):
        """
        Berechnet die Bild-Jacobi-Matrix (Interaktionsmatrix) L_s für einen Punkt.
        :param x: Normalisierte x-Koordinate
        :param y: Normalisierte y-Koordinate
        :param Z: Geschätzte Tiefe in Metern
        :return: 2x6 Numpy Array
        """
        L = np.zeros((2, 6))
        
        # Translation
        L[0, 0] = -1.0 / Z
        L[0, 1] = 0.0
        L[0, 2] = x / Z
        # Rotation
        L[0, 3] = x * y
        L[0, 4] = -(1.0 + x**2)
        L[0, 5] = y
        
        # Translation
        L[1, 0] = 0.0
        L[1, 1] = -1.0 / Z
        L[1, 2] = y / Z
        # Rotation
        L[1, 3] = 1.0 + y**2
        L[1, 4] = -x * y
        L[1, 5] = -x
        
        return L