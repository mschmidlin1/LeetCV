from __future__ import annotations

import numpy as np

def mean_reprojection_error(object_points: list[np.ndarray], image_points: list[np.ndarray], rvecs: list[np.ndarray], tvecs: list[np.ndarray], camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    raise NotImplementedError
