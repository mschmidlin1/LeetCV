import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q083"

from solutions.q083_mean_reprojection_error import mean_reprojection_error


class TestQ083(unittest.TestCase):
    def test_mean_reprojection_error(self):
        _object_points = np.load(CASE / 'object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        _rvecs = np.load(CASE / 'rvecs.npz')
        rvecs = [_rvecs[key] for key in sorted(_rvecs.files)]
        _tvecs = np.load(CASE / 'tvecs.npz')
        tvecs = [_tvecs[key] for key in sorted(_tvecs.files)]
        camera_matrix = np.load(CASE / 'camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'dist_coeffs.npy')
        actual = mean_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)


    def test_case_b(self):
        _object_points = np.load(CASE / 'case_b_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'case_b_image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        _rvecs = np.load(CASE / 'case_b_rvecs.npz')
        rvecs = [_rvecs[key] for key in sorted(_rvecs.files)]
        _tvecs = np.load(CASE / 'case_b_tvecs.npz')
        tvecs = [_tvecs[key] for key in sorted(_tvecs.files)]
        camera_matrix = np.load(CASE / 'case_b_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_b_dist_coeffs.npy')
        actual = mean_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)

    def test_case_c(self):
        _object_points = np.load(CASE / 'case_c_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'case_c_image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        _rvecs = np.load(CASE / 'case_c_rvecs.npz')
        rvecs = [_rvecs[key] for key in sorted(_rvecs.files)]
        _tvecs = np.load(CASE / 'case_c_tvecs.npz')
        tvecs = [_tvecs[key] for key in sorted(_tvecs.files)]
        camera_matrix = np.load(CASE / 'case_c_camera_matrix.npy')
        dist_coeffs = np.load(CASE / 'case_c_dist_coeffs.npy')
        actual = mean_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=1e-08)

if __name__ == "__main__":
    unittest.main()
