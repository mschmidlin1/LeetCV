import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q125"

from solutions.q125_pipeline import pipeline


class TestQ125(unittest.TestCase):
    def test_pipeline(self):
        self.assertIsNone(pipeline(str(ROOT / 'fixtures' / 'missing.png')))
        actual = pipeline(str(ROOT / 'fixtures' / 'dogs.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'expected.npy'))


    def test_case_b(self):
        self.assertIsNone(pipeline(str(CASE / 'missing-b.png')))

    def test_case_c(self):
        actual = pipeline(str(CASE / 'case_c.png'))
        np.testing.assert_array_equal(actual, np.load(CASE / 'case_c_expected.npy'))

if __name__ == "__main__":
    unittest.main()
