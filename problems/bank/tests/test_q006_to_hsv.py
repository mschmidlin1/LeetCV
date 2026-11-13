import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q006"

from solutions.q006_to_hsv import to_hsv, hsv_to_bgr


class TestQ006(unittest.TestCase):
    def test_to_hsv(self):
        image = np.load(CASE / 'image.npy')
        actual = to_hsv(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        hsv = np.load(CASE / 'hsv.npy')
        np.testing.assert_array_equal(hsv_to_bgr(hsv), np.load(CASE / 'expected_bgr.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = to_hsv(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        hsv = np.load(CASE / 'case_b_hsv.npy')
        np.testing.assert_array_equal(hsv_to_bgr(hsv), np.load(CASE / 'case_b_expected_bgr.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = to_hsv(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        hsv = np.load(CASE / 'case_c_hsv.npy')
        np.testing.assert_array_equal(hsv_to_bgr(hsv), np.load(CASE / 'case_c_expected_bgr.npy'))


if __name__ == "__main__":
    unittest.main()
