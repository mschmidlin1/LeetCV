import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q086"

from solutions.q086_stereo_disparity import stereo_disparity


class TestQ086(unittest.TestCase):
    def test_stereo_disparity(self):
        left = np.load(CASE / 'left.npy')
        right = np.load(CASE / 'right.npy')
        actual = stereo_disparity(left, right)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        left = np.load(CASE / 'case_b_left.npy')
        right = np.load(CASE / 'case_b_right.npy')
        actual = stereo_disparity(left, right)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        left = np.load(CASE / 'case_c_left.npy')
        right = np.load(CASE / 'case_c_right.npy')
        actual = stereo_disparity(left, right)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
