import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q104"

from solutions.q104_resize_frame import resize_frame


class TestQ104(unittest.TestCase):
    def test_resize_frame(self):
        frame = np.load(CASE / 'frame.npy')
        actual = resize_frame(frame, (20, 30))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        frame = np.load(CASE / 'case_b_frame.npy')
        actual = resize_frame(frame, (20, 30))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        frame = np.load(CASE / 'case_c_frame.npy')
        actual = resize_frame(frame, (20, 30))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
