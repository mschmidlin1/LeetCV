import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q118"

from solutions.q118_load_image import load_image


class TestQ118(unittest.TestCase):
    def test_load_image(self):
        actual = load_image(str(ROOT / 'fixtures' / 'dogs.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'expected.npy'))
        with self.assertRaises(FileNotFoundError):
            load_image(str(ROOT / 'fixtures' / 'missing.png'))


    def test_case_b(self):
        actual = load_image(str(CASE / 'case_b.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'case_b_expected.npy'))
        with self.assertRaises(FileNotFoundError):
            load_image(str(CASE / 'missing-b.png'))

    def test_case_c(self):
        actual = load_image(str(CASE / 'case_c.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'case_c_expected.npy'))

if __name__ == "__main__":
    unittest.main()
