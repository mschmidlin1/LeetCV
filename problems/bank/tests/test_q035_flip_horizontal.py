import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q035"

from solutions.q035_flip_horizontal import flip_horizontal, flip_vertical, flip_both


class TestQ035(unittest.TestCase):
    def test_flip_horizontal(self):
        image = np.load(CASE / 'image.npy')
        actual = flip_horizontal(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(flip_vertical(image), np.load(CASE / 'expected_vertical.npy'))
        np.testing.assert_array_equal(flip_both(image), np.load(CASE / 'expected_both.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = flip_horizontal(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(flip_vertical(image), np.load(CASE / 'case_b_expected_vertical.npy'))
        np.testing.assert_array_equal(flip_both(image), np.load(CASE / 'case_b_expected_both.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = flip_horizontal(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(flip_vertical(image), np.load(CASE / 'case_c_expected_vertical.npy'))
        np.testing.assert_array_equal(flip_both(image), np.load(CASE / 'case_c_expected_both.npy'))


if __name__ == "__main__":
    unittest.main()
