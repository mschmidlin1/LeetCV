import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q030"

from solutions.q030_orb_match_count import orb_match_count


class TestQ030(unittest.TestCase):
    def test_orb_match_count(self):
        image_a = np.load(CASE / 'image_a.npy')
        image_b = np.load(CASE / 'image_b.npy')
        actual = orb_match_count(image_a, image_b)
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


    def test_case_b(self):
        image_a = np.load(CASE / 'case_b_image_a.npy')
        image_b = np.load(CASE / 'case_b_image_b.npy')
        actual = orb_match_count(image_a, image_b)
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

    def test_case_c(self):
        image_a = np.load(CASE / 'case_c_image_a.npy')
        image_b = np.load(CASE / 'case_c_image_b.npy')
        actual = orb_match_count(image_a, image_b)
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


if __name__ == "__main__":
    unittest.main()
