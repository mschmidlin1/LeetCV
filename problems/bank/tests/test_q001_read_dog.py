import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q001"

from solutions.q001_read_dog import read_dog


class TestQ001(unittest.TestCase):
    def test_read_dog(self):
        actual = read_dog(str(ROOT / 'fixtures' / 'dogs.png'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        actual = read_dog(str(CASE / 'case_b.png'))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        actual = read_dog(str(CASE / 'case_c.png'))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
