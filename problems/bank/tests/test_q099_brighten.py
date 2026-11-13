import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q099"

from solutions.q099_brighten import brighten, wrap_add


class TestQ099(unittest.TestCase):
    def test_brighten(self):
        image = np.load(CASE / 'image.npy')
        actual = brighten(image, 40)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(wrap_add(image, 40), np.load(CASE / 'expected_wrap.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = brighten(image, 40)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(wrap_add(image, 40), np.load(CASE / 'case_b_expected_wrap.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = brighten(image, 40)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(wrap_add(image, 40), np.load(CASE / 'case_c_expected_wrap.npy'))


if __name__ == "__main__":
    unittest.main()
