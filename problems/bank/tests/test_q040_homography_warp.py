import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q040"

from solutions.q040_homography_warp import homography_warp


class TestQ040(unittest.TestCase):
    def test_homography_warp(self):
        source = np.load(CASE / 'source.npy')
        destination = np.load(CASE / 'destination.npy')
        actual = homography_warp(source, destination)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        source = np.load(CASE / 'case_b_source.npy')
        destination = np.load(CASE / 'case_b_destination.npy')
        actual = homography_warp(source, destination)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        source = np.load(CASE / 'case_c_source.npy')
        destination = np.load(CASE / 'case_c_destination.npy')
        actual = homography_warp(source, destination)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
