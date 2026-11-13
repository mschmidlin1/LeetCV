import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q018"

from solutions.q018_equalize_hist import equalize_hist, clahe


class TestQ018(unittest.TestCase):
    def test_equalize_hist(self):
        image = np.load(CASE / 'image.npy')
        actual = equalize_hist(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(clahe(image), np.load(CASE / 'expected_clahe.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = equalize_hist(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(clahe(image), np.load(CASE / 'case_b_expected_clahe.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = equalize_hist(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(clahe(image), np.load(CASE / 'case_c_expected_clahe.npy'))


if __name__ == "__main__":
    unittest.main()
