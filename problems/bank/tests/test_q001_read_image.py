import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q001"

from solutions.q001_read_image import read_16bit, read_color, read_greyscale


class TestQ001(unittest.TestCase):
    def test_read_image(self):
        actual = read_color(str(ROOT / 'fixtures' / 'dogs.png'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(read_greyscale(str(ROOT / 'fixtures' / 'dogs.png')), np.load(CASE / 'expected_grey.npy'))
        np.testing.assert_array_equal(read_16bit(str(CASE / 'sample_16.png')), np.load(CASE / 'sample_16_expected.npy'))

    def test_case_b(self):
        np.testing.assert_array_equal(read_color(str(CASE / 'case_b.png')), np.load(CASE / 'case_b_expected.npy'))
        np.testing.assert_array_equal(read_greyscale(str(CASE / 'case_b.png')), np.load(CASE / 'case_b_expected_grey.npy'))
        np.testing.assert_array_equal(read_16bit(str(CASE / 'sample_16_b.png')), np.load(CASE / 'sample_16_b_expected.npy'))

    def test_case_c(self):
        np.testing.assert_array_equal(read_color(str(CASE / 'case_c.png')), np.load(CASE / 'case_c_expected.npy'))
        np.testing.assert_array_equal(read_greyscale(str(CASE / 'case_c.png')), np.load(CASE / 'case_c_expected_grey.npy'))
        np.testing.assert_array_equal(read_16bit(str(CASE / 'sample_16_c.png')), np.load(CASE / 'sample_16_c_expected.npy'))

if __name__ == "__main__":
    unittest.main()
