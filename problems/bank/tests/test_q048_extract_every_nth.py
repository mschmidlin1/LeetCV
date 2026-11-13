import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q048"

from solutions.q048_extract_every_nth import extract_every_nth


class TestQ048(unittest.TestCase):
    def test_extract_every_nth(self):
        actual = extract_every_nth(str(ROOT / 'fixtures' / 'clip.avi'), 2)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        actual = extract_every_nth(str(CASE / 'case_b.avi'), 1)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        actual = extract_every_nth(str(CASE / 'case_c.avi'), 2)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
