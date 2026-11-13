import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q130"

from solutions.q130_inpaint_points import inpaint_points


class TestQ130(unittest.TestCase):
    def test_inpaint_points(self):
        image = np.load(CASE / 'image.npy')
        points = np.load(CASE / 'points.npy')
        actual = inpaint_points(image, points)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        points = np.load(CASE / 'case_b_points.npy')
        actual = inpaint_points(image, points)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        points = np.load(CASE / 'case_c_points.npy')
        actual = inpaint_points(image, points)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
