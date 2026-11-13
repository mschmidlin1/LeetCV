import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q026"

from solutions.q026_contour_count import contour_count, contour_all


class TestQ026(unittest.TestCase):
    def test_contour_count(self):
        image = np.load(CASE / 'image.npy')
        actual = contour_count(image)
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))
        expected_all = json.loads((CASE / 'expected_all.json').read_text(encoding='utf-8'))
        self.assertEqual(contour_all(image), expected_all)
        hole = np.load(CASE / 'hole.npy')
        self.assertEqual(contour_count(hole), json.loads((CASE / 'hole_external.json').read_text(encoding='utf-8')))
        self.assertEqual(contour_all(hole), json.loads((CASE / 'hole_all.json').read_text(encoding='utf-8')))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        actual = contour_count(image)
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))
        expected_all = json.loads((CASE / 'case_b_expected_all.json').read_text(encoding='utf-8'))
        self.assertEqual(contour_all(image), expected_all)
        hole = np.load(CASE / 'hole.npy')
        self.assertEqual(contour_count(hole), json.loads((CASE / 'hole_external.json').read_text(encoding='utf-8')))
        self.assertEqual(contour_all(hole), json.loads((CASE / 'hole_all.json').read_text(encoding='utf-8')))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        actual = contour_count(image)
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))
        expected_all = json.loads((CASE / 'case_c_expected_all.json').read_text(encoding='utf-8'))
        self.assertEqual(contour_all(image), expected_all)
        hole = np.load(CASE / 'hole.npy')
        self.assertEqual(contour_count(hole), json.loads((CASE / 'hole_external.json').read_text(encoding='utf-8')))
        self.assertEqual(contour_all(hole), json.loads((CASE / 'hole_all.json').read_text(encoding='utf-8')))


if __name__ == "__main__":
    unittest.main()
