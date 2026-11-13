import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q053"

from solutions.q053_detect_faces import detect_faces


class TestQ053(unittest.TestCase):
    def test_detect_faces(self):
        frames = np.load(CASE / 'frames.npy')
        actual = detect_faces(frames, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        frames = np.load(CASE / 'case_b_frames.npy')
        actual = detect_faces(frames, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        frames = np.load(CASE / 'case_c_frames.npy')
        actual = detect_faces(frames, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
