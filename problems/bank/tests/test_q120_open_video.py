import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q120"

from solutions.q120_open_video import open_video


class TestQ120(unittest.TestCase):
    def test_open_video(self):
        try:
            open_video(str(ROOT / 'fixtures' / 'missing.avi'))
        except RuntimeError as exc:
            if type(exc) is not RuntimeError:
                raise
        else:
            self.fail('RuntimeError not raised')


    def test_case_b(self):
        try:
            open_video(str(CASE / 'missing-b.avi'))
        except RuntimeError as exc:
            if type(exc) is not RuntimeError:
                raise
        else:
            self.fail('RuntimeError not raised')

    def test_case_c(self):
        capture = open_video(str(ROOT / 'fixtures' / 'clip.avi'))
        self.assertTrue(capture.isOpened())
        capture.release()

if __name__ == "__main__":
    unittest.main()
