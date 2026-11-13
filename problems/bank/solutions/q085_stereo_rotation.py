from __future__ import annotations

import numpy as np

def stereo_rotation(object_points: list[np.ndarray], image_points_left: list[np.ndarray], image_points_right: list[np.ndarray], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    raise NotImplementedError
