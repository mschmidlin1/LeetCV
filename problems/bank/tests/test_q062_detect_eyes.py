import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q062"

from solutions.q062_detect_eyes import detect_eyes


class TestQ062(unittest.TestCase):
    def test_detect_eyes(self):
        image = np.load(CASE / 'image.npy')
        actual = detect_eyes(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_eye.xml'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = detect_eyes(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_eye.xml'))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = detect_eyes(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_eye.xml'))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
