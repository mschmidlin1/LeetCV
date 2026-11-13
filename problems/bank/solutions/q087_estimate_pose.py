from __future__ import annotations

import numpy as np

def estimate_pose(object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
