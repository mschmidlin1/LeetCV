import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q043"

from solutions.q043_stabilize_frame import stabilize_frame


class TestQ043(unittest.TestCase):
    def test_stabilize_frame(self):
        previous = np.load(CASE / 'previous.npy')
        current = np.load(CASE / 'current.npy')
        actual = stabilize_frame(previous, current)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        previous = np.load(CASE / 'case_b_previous.npy')
        current = np.load(CASE / 'case_b_current.npy')
        actual = stabilize_frame(previous, current)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        previous = np.load(CASE / 'case_c_previous.npy')
        current = np.load(CASE / 'case_c_current.npy')
        actual = stabilize_frame(previous, current)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
