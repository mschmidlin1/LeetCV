import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q121"

from solutions.q121_blur_or_none import blur_or_none


class TestQ121(unittest.TestCase):
    def test_blur_or_none(self):
        self.assertIsNone(blur_or_none(None))
        image = np.load(CASE / 'image.npy')
        actual = blur_or_none(image)
        np.testing.assert_array_equal(actual, np.load(CASE / 'expected.npy'))


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = blur_or_none(image)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = blur_or_none(image)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
