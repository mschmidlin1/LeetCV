import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q055"

from solutions.q055_haar_faces import haar_faces


class TestQ055(unittest.TestCase):
    def test_haar_faces(self):
        image = np.load(CASE / 'image.npy')
        actual = haar_faces(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = haar_faces(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = haar_faces(image, str(ROOT / 'fixtures' / 'cascades' / 'haarcascade_frontalface_default.xml'))
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
