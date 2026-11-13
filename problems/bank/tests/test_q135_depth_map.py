import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q135"

from solutions.q135_depth_map import depth_map


class TestQ135(unittest.TestCase):
    def test_depth_map(self):
        left = np.load(CASE / 'left.npy')
        right = np.load(CASE / 'right.npy')
        actual = depth_map(left, right)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        left = np.load(CASE / 'case_b_left.npy')
        right = np.load(CASE / 'case_b_right.npy')
        actual = depth_map(left, right)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        left = np.load(CASE / 'case_c_left.npy')
        right = np.load(CASE / 'case_c_right.npy')
        actual = depth_map(left, right)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
