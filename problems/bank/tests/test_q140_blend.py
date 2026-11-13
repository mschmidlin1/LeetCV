import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q140"

from solutions.q140_blend import blend


class TestQ140(unittest.TestCase):
    def test_blend(self):
        image_a = np.load(CASE / 'image_a.npy')
        image_b = np.load(CASE / 'image_b.npy')
        actual = blend(image_a, image_b, 0.5)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image_a = np.load(CASE / 'case_b_image_a.npy')
        image_b = np.load(CASE / 'case_b_image_b.npy')
        actual = blend(image_a, image_b, 0.5)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image_a = np.load(CASE / 'case_c_image_a.npy')
        image_b = np.load(CASE / 'case_c_image_b.npy')
        actual = blend(image_a, image_b, 0.5)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
