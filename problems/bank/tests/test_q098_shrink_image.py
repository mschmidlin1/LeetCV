import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q098"

from solutions.q098_shrink_image import shrink_image


class TestQ098(unittest.TestCase):
    def test_shrink_image(self):
        image = np.load(CASE / 'image.npy')
        actual = shrink_image(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = shrink_image(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = shrink_image(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
