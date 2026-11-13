import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q103"

from solutions.q103_parallel_orb_counts import parallel_orb_counts


class TestQ103(unittest.TestCase):
    def test_parallel_orb_counts(self):
        images = np.load(CASE / 'images.npy')
        actual = parallel_orb_counts(images)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        images = np.load(CASE / 'case_b_images.npy')
        actual = parallel_orb_counts(images)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        images = np.load(CASE / 'case_c_images.npy')
        actual = parallel_orb_counts(images)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
