import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q131"

from solutions.q131_project_origin import project_origin


class TestQ131(unittest.TestCase):
    def test_project_origin(self):
        rvec = np.load(CASE / 'rvec.npy')
        tvec = np.load(CASE / 'tvec.npy')
        camera_matrix = np.load(CASE / 'camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'dist_coeffs.npy')
        actual = project_origin(rvec, tvec, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        rvec = np.load(CASE / 'case_b_rvec.npy')
        tvec = np.load(CASE / 'case_b_tvec.npy')
        camera_matrix = np.load(CASE / 'case_b_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_b_dist_coeffs.npy')
        actual = project_origin(rvec, tvec, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        rvec = np.load(CASE / 'case_c_rvec.npy')
        tvec = np.load(CASE / 'case_c_tvec.npy')
        camera_matrix = np.load(CASE / 'case_c_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_c_dist_coeffs.npy')
        actual = project_origin(rvec, tvec, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
