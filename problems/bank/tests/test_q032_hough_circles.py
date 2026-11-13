import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q032"

from solutions.q032_hough_circles import hough_circles


class TestQ032(unittest.TestCase):
    def test_hough_circles(self):
        image = np.load(CASE / 'image.npy')
        actual = hough_circles(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = hough_circles(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = hough_circles(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
