import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q041"

from solutions.q041_farneback_flow import farneback_flow


class TestQ041(unittest.TestCase):
    def test_farneback_flow(self):
        first = np.load(CASE / 'first.npy')
        second = np.load(CASE / 'second.npy')
        actual = farneback_flow(first, second)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        first = np.load(CASE / 'case_b_first.npy')
        second = np.load(CASE / 'case_b_second.npy')
        actual = farneback_flow(first, second)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        first = np.load(CASE / 'case_c_first.npy')
        second = np.load(CASE / 'case_c_second.npy')
        actual = farneback_flow(first, second)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
