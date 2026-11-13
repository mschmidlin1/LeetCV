import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q050"

from solutions.q050_grayscale_video import grayscale_video


class TestQ050(unittest.TestCase):
    def test_grayscale_video(self):
        folder = tempfile.mkdtemp()
        output_path = str(Path(folder) / 'out.avi')
        actual = grayscale_video(str(ROOT / 'fixtures' / 'clip.avi'), output_path)
        self.assertEqual(Path(output_path).read_bytes(), (CASE / 'expected.bin').read_bytes())


    def test_case_b(self):
        import cv2
        folder = tempfile.mkdtemp()
        output_path = str(Path(folder) / 'out.avi')
        grayscale_video(str(CASE / 'case_b.avi'), output_path)
        capture = cv2.VideoCapture(output_path)
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0].shape[0], 16)
        self.assertEqual(decoded[0].shape[1], 24)

    def test_case_c(self):
        import cv2
        folder = tempfile.mkdtemp()
        output_path = str(Path(folder) / 'out.avi')
        grayscale_video(str(CASE / 'case_c.avi'), output_path)
        capture = cv2.VideoCapture(output_path)
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
        self.assertEqual(len(decoded), 3)
        self.assertEqual(decoded[0].shape[0], 12)
        self.assertEqual(decoded[0].shape[1], 18)

if __name__ == "__main__":
    unittest.main()
