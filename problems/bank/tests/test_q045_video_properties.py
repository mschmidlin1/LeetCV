import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q045"

from solutions.q045_video_properties import video_properties


class TestQ045(unittest.TestCase):
    def test_video_properties(self):
        actual = video_properties(str(ROOT / 'fixtures' / 'clip.avi'))
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


    def test_case_b(self):
        actual = video_properties(str(CASE / 'case_b.avi'))
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

    def test_case_c(self):
        actual = video_properties(str(CASE / 'case_c.avi'))
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

if __name__ == "__main__":
    unittest.main()
