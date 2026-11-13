import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q087"

from solutions.q087_estimate_pose import estimate_pose


class TestQ087(unittest.TestCase):
    def test_estimate_pose(self):
        object_points = np.load(CASE / 'object_points.npy')
        image_points = np.load(CASE / 'image_points.npy')
        camera_matrix = np.load(CASE / 'camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'dist_coeffs.npy')
        actual = estimate_pose(object_points, image_points, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        object_points = np.load(CASE / 'case_b_object_points.npy')
        image_points = np.load(CASE / 'case_b_image_points.npy')
        camera_matrix = np.load(CASE / 'case_b_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_b_dist_coeffs.npy')
        actual = estimate_pose(object_points, image_points, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_b_expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        object_points = np.load(CASE / 'case_c_object_points.npy')
        image_points = np.load(CASE / 'case_c_image_points.npy')
        camera_matrix = np.load(CASE / 'case_c_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_c_dist_coeffs.npy')
        actual = estimate_pose(object_points, image_points, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_c_expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
