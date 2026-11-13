import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q013"

from solutions.q013_binary_threshold import binary_threshold, binary_inv, otsu_threshold


class TestQ013(unittest.TestCase):
    def test_binary_threshold(self):
        image = np.load(CASE / 'image.npy')
        actual = binary_threshold(image, 127)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(binary_inv(image, 127), np.load(CASE / 'expected_inv.npy'))
        np.testing.assert_array_equal(otsu_threshold(image), np.load(CASE / 'expected_otsu.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = binary_threshold(image, 127)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(binary_inv(image, 127), np.load(CASE / 'case_b_expected_inv.npy'))
        np.testing.assert_array_equal(otsu_threshold(image), np.load(CASE / 'case_b_expected_otsu.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = binary_threshold(image, 127)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(binary_inv(image, 127), np.load(CASE / 'case_c_expected_inv.npy'))
        np.testing.assert_array_equal(otsu_threshold(image), np.load(CASE / 'case_c_expected_otsu.npy'))


if __name__ == "__main__":
    unittest.main()
