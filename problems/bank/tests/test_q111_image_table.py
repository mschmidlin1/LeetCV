import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q111"

from solutions.q111_image_table import image_table


class TestQ111(unittest.TestCase):
    def test_image_table(self):
        paths = [
            str(ROOT / 'fixtures' / 'dogs.png'),
            str(ROOT / 'fixtures' / 'images' / 'shapes.png'),
        ]
        actual = image_table(paths)
        self.assertEqual(list(actual.columns), ["path", "width", "height"])
        self.assertEqual(actual['path'].tolist(), paths)
        self.assertEqual(actual['width'].tolist(), [48, 80])
        self.assertEqual(actual['height'].tolist(), [48, 80])


    def test_case_b(self):
        import cv2
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'pic.png')
        cv2.imwrite(path, np.zeros((15, 21, 3), np.uint8))
        actual = image_table([path])
        self.assertEqual(list(actual.columns), ["path", "width", "height"])
        self.assertEqual(actual["path"].tolist(), [path])
        self.assertEqual(actual["width"].tolist(), [21])
        self.assertEqual(actual["height"].tolist(), [15])

    def test_case_c(self):
        import cv2
        folder = tempfile.mkdtemp()
        first = str(Path(folder) / 'a.png')
        second = str(Path(folder) / 'b.png')
        cv2.imwrite(first, np.zeros((9, 13, 3), np.uint8))
        cv2.imwrite(second, np.zeros((30, 10, 3), np.uint8))
        actual = image_table([first, second])
        self.assertEqual(actual["width"].tolist(), [13, 10])
        self.assertEqual(actual["height"].tolist(), [9, 30])

if __name__ == "__main__":
    unittest.main()
