import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q137"

from solutions.q137_sobel_gradients import sobel_gradients


class TestQ137(unittest.TestCase):
    def test_sobel_gradients(self):
        image = np.load(CASE / 'image.npy')
        actual = sobel_gradients(image)
        expected = np.load(CASE / 'expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = sobel_gradients(image)
        expected = np.load(CASE / 'case_b_expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = sobel_gradients(image)
        expected = np.load(CASE / 'case_c_expected.npz')
        first, second = actual
        np.testing.assert_allclose(first, expected['first'], rtol=1e-05, atol=0.0001)
        np.testing.assert_allclose(second, expected['second'], rtol=1e-05, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
