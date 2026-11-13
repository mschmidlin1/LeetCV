import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q067"

from solutions.q067_track_last_box import track_last_box


class TestQ067(unittest.TestCase):
    def test_track_last_box(self):
        frames = np.load(CASE / 'frames.npy')
        actual = track_last_box(frames)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=0.001, atol=0.01)


    def test_case_b(self):
        frames = np.load(CASE / 'case_b_frames.npy')
        actual = track_last_box(frames)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=0.001, atol=0.01)

    def test_case_c(self):
        frames = np.load(CASE / 'case_c_frames.npy')
        actual = track_last_box(frames)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_allclose(actual, expected, rtol=0.001, atol=0.01)


if __name__ == "__main__":
    unittest.main()
