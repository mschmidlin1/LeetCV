import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q044"

from solutions.q044_read_frames import read_frames


class TestQ044(unittest.TestCase):
    def test_read_frames(self):
        actual = read_frames(str(ROOT / 'fixtures' / 'clip.avi'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        actual = read_frames(str(CASE / 'case_b.avi'))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        actual = read_frames(str(CASE / 'case_c.avi'))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
