import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q100"

from solutions.q100_blur_frames import blur_frames


class TestQ100(unittest.TestCase):
    def test_blur_frames(self):
        frames = np.load(CASE / 'frames.npy')
        actual = blur_frames(frames)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        frames = np.load(CASE / 'case_b_frames.npy')
        actual = blur_frames(frames)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        frames = np.load(CASE / 'case_c_frames.npy')
        actual = blur_frames(frames)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
