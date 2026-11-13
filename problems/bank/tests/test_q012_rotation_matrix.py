import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q012"

from solutions.q012_rotation_matrix import rotation_matrix


class TestQ012(unittest.TestCase):
    def test_rotation_matrix(self):
        actual = rotation_matrix((64, 64), 45.0)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        actual = rotation_matrix((32, 48), 90.0)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        actual = rotation_matrix((20, 20), -15.0)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=1e-05, atol=0.0001)

if __name__ == "__main__":
    unittest.main()
