import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q123"

from solutions.q123_read_frame_with_retry import read_frame_with_retry


class TestQ123(unittest.TestCase):
    def test_read_frame_with_retry(self):
        class FakeCap:
            def __init__(self):
                self.calls = 0

            def read(self):
                self.calls += 1
                if self.calls < 3:
                    return False, None
                return True, np.zeros((8, 8, 3), np.uint8)

        actual = read_frame_with_retry(FakeCap(), 5)
        np.testing.assert_array_equal(actual, np.zeros((8, 8, 3), np.uint8))


    def test_case_b(self):
        class FakeCap:
            def read(self):
                return True, np.full((4, 6, 3), 7, np.uint8)

        actual = read_frame_with_retry(FakeCap(), 2)
        np.testing.assert_array_equal(actual, np.full((4, 6, 3), 7, np.uint8))

    def test_case_c(self):
        class FakeCap:
            def read(self):
                return False, None

        with self.assertRaises(RuntimeError):
            read_frame_with_retry(FakeCap(), 3)

if __name__ == "__main__":
    unittest.main()
