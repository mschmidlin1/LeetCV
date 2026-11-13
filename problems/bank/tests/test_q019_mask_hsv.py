import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q019"

from solutions.q019_mask_hsv import mask_hsv


class TestQ019(unittest.TestCase):
    def test_mask_hsv(self):
        image = np.load(CASE / 'image.npy')
        lower = np.load(CASE / 'lower.npy')
        upper = np.load(CASE / 'upper.npy')
        actual = mask_hsv(image, lower, upper)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        lower = np.load(CASE / 'case_b_lower.npy')
        upper = np.load(CASE / 'case_b_upper.npy')
        actual = mask_hsv(image, lower, upper)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        lower = np.load(CASE / 'case_c_lower.npy')
        upper = np.load(CASE / 'case_c_upper.npy')
        actual = mask_hsv(image, lower, upper)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
