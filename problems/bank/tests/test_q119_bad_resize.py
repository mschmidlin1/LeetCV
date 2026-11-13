import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q119"

from solutions.q119_bad_resize import bad_resize


class TestQ119(unittest.TestCase):
    def test_bad_resize(self):
        import cv2
        image = np.load(CASE / 'image.npy')
        with self.assertRaises(cv2.error):
            bad_resize(image)


    def test_case_b(self):
        import cv2
        image = np.zeros((8, 9, 3), np.uint8)
        with self.assertRaises(cv2.error):
            bad_resize(image)

    def test_case_c(self):
        import cv2
        image = np.full((5, 7), 3, np.uint8)
        with self.assertRaises(cv2.error):
            bad_resize(image)

if __name__ == "__main__":
    unittest.main()
