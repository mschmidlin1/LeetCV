import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q127"

from solutions.q127_inpaint import inpaint


class TestQ127(unittest.TestCase):
    def test_inpaint(self):
        image = np.load(CASE / 'image.npy')
        mask = np.load(CASE / 'mask.npy')
        actual = inpaint(image, mask)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        mask = np.load(CASE / 'case_b_mask.npy')
        actual = inpaint(image, mask)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        mask = np.load(CASE / 'case_c_mask.npy')
        actual = inpaint(image, mask)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
