import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q046"

from solutions.q046_write_video import write_video


class TestQ046(unittest.TestCase):
    def test_write_video(self):
        frames = np.load(CASE / 'frames.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.avi')
        actual = write_video(frames, path)
        self.assertEqual(Path(path).read_bytes(), (CASE / 'expected.bin').read_bytes())


    def test_case_b(self):
        import cv2
        frames = np.load(CASE / 'case_b_frames.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.avi')
        write_video(frames, path)
        capture = cv2.VideoCapture(path)
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
        self.assertEqual(len(decoded), frames.shape[0])
        self.assertEqual(decoded[0].shape[0], frames.shape[1])
        self.assertEqual(decoded[0].shape[1], frames.shape[2])

    def test_case_c(self):
        import cv2
        frames = np.load(CASE / 'case_c_frames.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.avi')
        write_video(frames, path)
        capture = cv2.VideoCapture(path)
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        capture.release()
        self.assertEqual(len(decoded), frames.shape[0])
        self.assertEqual(decoded[0].shape[0], frames.shape[1])
        self.assertEqual(decoded[0].shape[1], frames.shape[2])

if __name__ == "__main__":
    unittest.main()
