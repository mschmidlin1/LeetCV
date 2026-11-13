import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q028"

from solutions.q028_sift_keypoint_count import sift_keypoint_count


class TestQ028(unittest.TestCase):
    def test_sift_keypoint_count(self):
        image = np.load(CASE / 'image.npy')
        actual = sift_keypoint_count(image)
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = sift_keypoint_count(image)
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = sift_keypoint_count(image)
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


if __name__ == "__main__":
    unittest.main()
