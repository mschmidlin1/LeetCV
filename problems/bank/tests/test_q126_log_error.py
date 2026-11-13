import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q126"

from solutions.q126_log_error import log_error


class TestQ126(unittest.TestCase):
    def test_log_error(self):
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'errors.log')
        log_error(path, 'boom')
        self.assertEqual(Path(path).read_text(encoding='utf-8'), 'boom\n')


    def test_case_b(self):
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'errors.log')
        log_error(path, 'first')
        log_error(path, 'second')
        self.assertEqual(Path(path).read_text(encoding='utf-8'), 'first\nsecond\n')

    def test_case_c(self):
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'errors.log')
        log_error(path, '')
        self.assertEqual(Path(path).read_text(encoding='utf-8'), '\n')

if __name__ == "__main__":
    unittest.main()
