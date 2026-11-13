import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q084"

from solutions.q084_calibrate_views import calibrate_views


class TestQ084(unittest.TestCase):
    def test_calibrate_views(self):
        _object_points = np.load(CASE / 'object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        actual = calibrate_views(object_points, image_points, (160, 120))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        _object_points = np.load(CASE / 'case_b_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'case_b_image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        actual = calibrate_views(object_points, image_points, (160, 120))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        _object_points = np.load(CASE / 'case_c_object_points.npz')
        object_points = [_object_points[key] for key in sorted(_object_points.files)]
        _image_points = np.load(CASE / 'case_c_image_points.npz')
        image_points = [_image_points[key] for key in sorted(_image_points.files)]
        actual = calibrate_views(object_points, image_points, (160, 120))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

if __name__ == "__main__":
    unittest.main()
