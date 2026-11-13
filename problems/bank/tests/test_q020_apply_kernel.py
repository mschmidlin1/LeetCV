import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q020"

from solutions.q020_apply_kernel import apply_kernel


class TestQ020(unittest.TestCase):
    def test_apply_kernel(self):
        image = np.load(CASE / 'image.npy')
        kernel = np.load(CASE / 'kernel.npy')
        actual = apply_kernel(image, kernel)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        kernel = np.load(CASE / 'case_b_kernel.npy')
        actual = apply_kernel(image, kernel)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        kernel = np.load(CASE / 'case_c_kernel.npy')
        actual = apply_kernel(image, kernel)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
