import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q021"

from solutions.q021_dilate_image import dilate_image, erode_image, morph_open


class TestQ021(unittest.TestCase):
    def test_dilate_image(self):
        image = np.load(CASE / 'image.npy')
        actual = dilate_image(image)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(erode_image(image), np.load(CASE / 'expected_erode.npy'))
        np.testing.assert_array_equal(morph_open(image), np.load(CASE / 'expected_open.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = dilate_image(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(erode_image(image), np.load(CASE / 'case_b_expected_erode.npy'))
        np.testing.assert_array_equal(morph_open(image), np.load(CASE / 'case_b_expected_open.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = dilate_image(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(erode_image(image), np.load(CASE / 'case_c_expected_erode.npy'))
        np.testing.assert_array_equal(morph_open(image), np.load(CASE / 'case_c_expected_open.npy'))


if __name__ == "__main__":
    unittest.main()
