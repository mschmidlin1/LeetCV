import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q052"

from solutions.q052_foreground_mask import foreground_mask


class TestQ052(unittest.TestCase):
    def test_foreground_mask(self):
        frames = np.load(CASE / 'frames.npy')
        actual = foreground_mask(frames)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        frames = np.load(CASE / 'case_b_frames.npy')
        actual = foreground_mask(frames)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        frames = np.load(CASE / 'case_c_frames.npy')
        actual = foreground_mask(frames)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
