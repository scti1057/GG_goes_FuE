import numpy as np
import cv2

class BaseFilter:
    """
    Zentrale Basisklasse für alle Image-Space Filter.
    Übernimmt Keypoint-Auswahl, Messaufbereitung und Geometrie-Check.
    """
    def __init__(self, K):
        self.K = K
        self.initialized = False
        self.status = "INIT"
        self.H_filtered = np.eye(3, dtype=np.float64)
        self.active_ref_ids = np.zeros((0,), dtype=np.int64)
        self.active_desired = np.zeros((2, 0), dtype=np.float64)
        self.max_active_keypoints = 80
        self.min_init_keypoints = 8
        self.min_update_keypoints = 4

    def reset(self):
        self.initialized = False
        self.status = "INIT"
        self.H_filtered = np.eye(3, dtype=np.float64)
        self.active_ref_ids = np.zeros((0,), dtype=np.int64)
        self.active_desired = np.zeros((2, 0), dtype=np.float64)

    def force_relocalization(self):
        self.initialized = False
        self.status = "RELOCALIZING"
        self.active_ref_ids = np.zeros((0,), dtype=np.int64)
        self.active_desired = np.zeros((2, 0), dtype=np.float64)
        self.H_filtered = np.eye(3, dtype=np.float64)

    def configure_keypoint_tracking(
        self,
        max_active_keypoints,
        min_init_keypoints,
        min_update_keypoints,
    ):
        self.max_active_keypoints = max(4, int(max_active_keypoints))
        self.min_init_keypoints = max(4, int(min_init_keypoints))
        self.min_update_keypoints = max(1, int(min_update_keypoints))
        self.min_init_keypoints = min(
            self.min_init_keypoints, self.max_active_keypoints
        )
        self.min_update_keypoints = min(
            self.min_update_keypoints, self.max_active_keypoints
        )

    def get_active_count(self):
        return int(self.active_ref_ids.size)

    def _select_spread_indices(self, desired_pixels, match_scores, max_count):
        m = desired_pixels.shape[1]
        if m <= max_count:
            return np.arange(m, dtype=np.int64)

        if match_scores is not None and match_scores.size == m:
            scores = np.asarray(match_scores, dtype=np.float64)
            if np.isfinite(scores).all():
                s_min = float(np.min(scores))
                s_max = float(np.max(scores))
                if s_max > s_min:
                    scores = (scores - s_min) / (s_max - s_min)
                else:
                    scores = np.zeros((m,), dtype=np.float64)
            else:
                scores = np.zeros((m,), dtype=np.float64)
        else:
            scores = np.zeros((m,), dtype=np.float64)

        centroid = np.mean(desired_pixels, axis=1, keepdims=True)
        dist_to_centroid = np.sum((desired_pixels - centroid) ** 2, axis=0)
        if np.max(scores) > 0.0:
            first_idx = int(np.argmax(scores))
        else:
            first_idx = int(np.argmax(dist_to_centroid))

        selected = [first_idx]
        min_dist2 = np.full((m,), np.inf, dtype=np.float64)
        score_weight = 0.05

        for _ in range(1, max_count):
            last_idx = selected[-1]
            delta = desired_pixels - desired_pixels[:, last_idx:last_idx + 1]
            dist2 = np.sum(delta * delta, axis=0)
            min_dist2 = np.minimum(min_dist2, dist2)
            finite_min = min_dist2[np.isfinite(min_dist2)]
            scale = float(np.max(finite_min)) if finite_min.size > 0 else 1.0
            objective = min_dist2 + score_weight * scale * scores
            objective[selected] = -np.inf
            next_idx = int(np.argmax(objective))
            if not np.isfinite(objective[next_idx]):
                break
            selected.append(next_idx)

        return np.asarray(selected, dtype=np.int64)

    def _initialize_active_set(self, current_pixels, desired_pixels, ref_ids, match_scores):
        m = desired_pixels.shape[1]
        if m <= 0:
            return None

        # Doppelte Referenz-IDs entfernen (bevorzugt höhere Similarity).
        has_scores = (match_scores is not None) and (match_scores.size == m)
        best_per_id = {}
        for i in range(m):
            rid = int(ref_ids[i])
            sc = float(match_scores[i]) if has_scores else 0.0
            prev = best_per_id.get(rid)
            if (prev is None) or (sc > prev[0]):
                best_per_id[rid] = (sc, i)
        unique_indices = sorted(v[1] for v in best_per_id.values())
        if len(unique_indices) < m:
            keep = np.asarray(unique_indices, dtype=np.int64)
            current_pixels = current_pixels[:, keep]
            desired_pixels = desired_pixels[:, keep]
            ref_ids = ref_ids[keep]
            if has_scores:
                match_scores = match_scores[keep]
            m = desired_pixels.shape[1]

        if m < self.min_init_keypoints:
            return None

        selected = self._select_spread_indices(
            desired_pixels,
            match_scores,
            min(self.max_active_keypoints, m),
        )
        self.active_ref_ids = np.asarray(ref_ids[selected], dtype=np.int64)
        self.active_desired = desired_pixels[:, selected].astype(np.float64)

        # Re-lokalisierung verlangt eine vollständige Neuinitialisierung des Zustands.
        self.initialized = False
        self.H_filtered = np.eye(3, dtype=np.float64)
        return current_pixels[:, selected].astype(np.float64)

    @staticmethod
    def _stack_measurement(current_obs):
        m = current_obs.shape[1]
        z_k = np.zeros((2 * m, 1), dtype=np.float64)
        z_k[0::2, 0] = current_obs[0, :]
        z_k[1::2, 0] = current_obs[1, :]
        return z_k

    def _prepare_measurement(self, current_pixels, desired_pixels, ref_ids=None, match_scores=None):
        if current_pixels is None or desired_pixels is None:
            return None, None, "MISSING"
        if current_pixels.ndim != 2 or desired_pixels.ndim != 2:
            return None, None, "MISSING"
        if current_pixels.shape[0] != 2 or desired_pixels.shape[0] != 2:
            return None, None, "MISSING"

        m = min(current_pixels.shape[1], desired_pixels.shape[1])
        if m <= 0:
            return None, None, "MISSING"
        current_pixels = np.asarray(current_pixels[:, :m], dtype=np.float64)
        desired_pixels = np.asarray(desired_pixels[:, :m], dtype=np.float64)

        if ref_ids is None:
            ref_ids = np.arange(m, dtype=np.int64)
        else:
            ref_ids = np.asarray(ref_ids[:m], dtype=np.int64)
        if match_scores is None:
            match_scores = np.zeros((m,), dtype=np.float64)
        else:
            match_scores = np.asarray(match_scores[:m], dtype=np.float64)

        if self.active_ref_ids.size <= 0:
            current_selected = self._initialize_active_set(
                current_pixels,
                desired_pixels,
                ref_ids,
                match_scores,
            )
            if current_selected is None:
                return None, None, "MISSING"
            obs_slots = np.arange(self.get_active_count(), dtype=np.int64)
            return self._stack_measurement(current_selected), obs_slots, "INIT SET"

        id_to_slot = {int(ref_id): idx for idx, ref_id in enumerate(self.active_ref_ids.tolist())}
        seen_slots = set()
        obs_slots = []
        obs_points = []
        for i in range(m):
            slot = id_to_slot.get(int(ref_ids[i]))
            if slot is None or slot in seen_slots:
                continue
            seen_slots.add(slot)
            obs_slots.append(slot)
            obs_points.append(current_pixels[:, i])

        if len(obs_slots) < self.min_update_keypoints:
            current_selected = self._initialize_active_set(
                current_pixels,
                desired_pixels,
                ref_ids,
                match_scores,
            )
            if current_selected is None:
                return None, None, "MISSING"
            obs_slots = np.arange(self.get_active_count(), dtype=np.int64)
            return self._stack_measurement(current_selected), obs_slots, "RELOCALIZED"

        obs_slots = np.asarray(obs_slots, dtype=np.int64)
        current_obs = np.asarray(obs_points, dtype=np.float64).T
        return self._stack_measurement(current_obs), obs_slots, self.status

    def _check_geometry_and_update_H(self, state_8d, log_on_fail=True):
        n = self.get_active_count()
        if n < 4:
            return False

        state_vec = np.asarray(state_8d, dtype=np.float64).reshape(-1, 1)
        if state_vec.shape[0] != (2 * n):
            return False

        pts_cur = state_vec.reshape(n, 2).astype(np.float32)
        pts_des = self.active_desired.T.astype(np.float32)

        H, _ = cv2.findHomography(pts_des, pts_cur, method=0)
        if H is None or (not np.isfinite(H).all()):
            if log_on_fail:
                print("\n[Filter WARNING] Geometry update failed (homography invalid).")
                print(20*"-")
            return False

        if abs(float(H[2, 2])) > 1e-12:
            H = H / H[2, 2]
        self.H_filtered = H.astype(np.float64)
        return True

    def update_geometry_from_state(self, state_8d, log_on_fail=False):
        """Update H_filtered from a predicted 2N state."""
        if self.get_active_count() < 4:
            return False
        state_8d = np.asarray(state_8d, dtype=np.float64).reshape(-1, 1)
        return self._check_geometry_and_update_H(state_8d, log_on_fail=log_on_fail)

    def get_projected_points(self, desired_features):
        if not self.initialized or self.H_filtered is None:
            return desired_features
        M = desired_features.shape[1]
        hom_pts = np.vstack((desired_features, np.ones((1, M))))
        proj_hom = self.H_filtered @ hom_pts
        return np.vstack((proj_hom[0, :] / proj_hom[2, :], proj_hom[1, :] / proj_hom[2, :]))

    def get_active_ref_ids(self):
        return self.active_ref_ids.copy()

    def get_active_filtered_points(self):
        n = self.get_active_count()
        if (not self.initialized) or n <= 0:
            return np.zeros((2, 0), dtype=np.float64)
        state_vec = self._get_active_state_vector()
        state_vec = np.asarray(state_vec, dtype=np.float64).reshape(-1, 1)
        if state_vec.shape[0] != (2 * n):
            return np.zeros((2, 0), dtype=np.float64)
        return state_vec.reshape(n, 2).T.copy()

    def _transform_twist_ee_to_cam(self, v_ee):
        """Map project-specific EE twist components into the camera frame.

        The current convention used by the IBVS filters is:
        [vx, vy, vz, wx, wy, wz] -> [-vx, +vy, +vz, -wx, -wy, -wz]
        """
        v_ee = np.asarray(v_ee, dtype=np.float64).reshape(6,)
        return np.array([
            -v_ee[0],
             v_ee[1],
             v_ee[2],
             v_ee[3],
             v_ee[4],
            -v_ee[5],
        ], dtype=np.float64)

    def predict(self, v_ee, Z_est, dt): raise NotImplementedError
    def _get_active_state_vector(self): raise NotImplementedError
    def update(self, current_pixels, desired_pixels, ref_ids=None, match_scores=None): raise NotImplementedError
