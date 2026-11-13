import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q058"

from solutions.q058_template_top_left import template_top_left


class TestQ058(unittest.TestCase):
    def test_template_top_left(self):
        image = np.load(CASE / 'image.npy')
        template = np.load(CASE / 'template.npy')
        actual = template_top_left(image, template)
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        template = np.load(CASE / 'case_b_template.npy')
        actual = template_top_left(image, template)
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        template = np.load(CASE / 'case_c_template.npy')
        actual = template_top_left(image, template)
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


if __name__ == "__main__":
    unittest.main()
