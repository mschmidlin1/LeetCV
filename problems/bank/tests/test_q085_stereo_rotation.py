import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q085"

from solutions.q085_stereo_rotation import stereo_rotation


class TestQ085(unittest.TestCase):
    def test_stereo_rotation(self):
        _object_points = np.load(CASE / 'object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points_left = np.load(CASE / 'image_points_left.npz')
        image_points_left = [_image_points_left[key] for key in sorted(_image_points_left.files)]
        _image_points_right = np.load(CASE / 'image_points_right.npz')
        image_points_right = [_image_points_right[key] for key in sorted(_image_points_right.files)]
        camera_matrix = np.load(CASE / 'camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'dist_coeffs.npy')
        actual = stereo_rotation(object_points, image_points_left, image_points_right, camera_matrix, dist_coeffs, (160, 120))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        _object_points = np.load(CASE / 'case_b_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points_left = np.load(CASE / 'case_b_image_points_left.npz')
        image_points_left = [_image_points_left[key] for key in sorted(_image_points_left.files)]
        _image_points_right = np.load(CASE / 'case_b_image_points_right.npz')
        image_points_right = [_image_points_right[key] for key in sorted(_image_points_right.files)]
        camera_matrix = np.load(CASE / 'case_b_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_b_dist_coeffs.npy')
        actual = stereo_rotation(object_points, image_points_left, image_points_right, camera_matrix, dist_coeffs, (160, 120))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        _object_points = np.load(CASE / 'case_c_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points_left = np.load(CASE / 'case_c_image_points_left.npz')
        image_points_left = [_image_points_left[key] for key in sorted(_image_points_left.files)]
        _image_points_right = np.load(CASE / 'case_c_image_points_right.npz')
        image_points_right = [_image_points_right[key] for key in sorted(_image_points_right.files)]
        camera_matrix = np.load(CASE / 'case_c_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_c_dist_coeffs.npy')
        actual = stereo_rotation(object_points, image_points_left, image_points_right, camera_matrix, dist_coeffs, (160, 120))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

if __name__ == "__main__":
    unittest.main()
