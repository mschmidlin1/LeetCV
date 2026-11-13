import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q082"

from solutions.q082_undistort_image import undistort_image


class TestQ082(unittest.TestCase):
    def test_undistort_image(self):
        image = np.load(CASE / 'image.npy')
        camera_matrix = np.load(CASE / 'camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'dist_coeffs.npy')
        actual = undistort_image(image, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        camera_matrix = np.load(CASE / 'case_b_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_b_dist_coeffs.npy')
        actual = undistort_image(image, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        camera_matrix = np.load(CASE / 'case_c_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_c_dist_coeffs.npy')
        actual = undistort_image(image, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
