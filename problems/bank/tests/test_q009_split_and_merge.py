import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q009"

from solutions.q009_split_and_merge import split_and_merge, swap_rb, blue_channel


class TestQ009(unittest.TestCase):
    def test_split_and_merge(self):
        image = np.load(CASE / 'image.npy')
        actual = split_and_merge(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(swap_rb(image), np.load(CASE / 'expected_swap.npy'))
        np.testing.assert_array_equal(blue_channel(image), np.load(CASE / 'expected_blue.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = split_and_merge(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(swap_rb(image), np.load(CASE / 'case_b_expected_swap.npy'))
        np.testing.assert_array_equal(blue_channel(image), np.load(CASE / 'case_b_expected_blue.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = split_and_merge(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(swap_rb(image), np.load(CASE / 'case_c_expected_swap.npy'))
        np.testing.assert_array_equal(blue_channel(image), np.load(CASE / 'case_c_expected_blue.npy'))


if __name__ == "__main__":
    unittest.main()
