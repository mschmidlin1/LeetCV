import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q124"

from solutions.q124_load_or_raise import OpenCVError, load_or_raise


class TestQ124(unittest.TestCase):
    def test_load_or_raise(self):
        actual = load_or_raise(str(ROOT / 'fixtures' / 'dogs.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'expected.npy'))
        with self.assertRaises(OpenCVError):
            load_or_raise(str(ROOT / 'fixtures' / 'missing.png'))


    def test_case_b(self):
        actual = load_or_raise(str(CASE / 'case_b.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'case_b_expected.npy'))

    def test_case_c(self):
        with self.assertRaises(OpenCVError):
            load_or_raise(str(CASE / 'missing-c.png'))

if __name__ == "__main__":
    unittest.main()
