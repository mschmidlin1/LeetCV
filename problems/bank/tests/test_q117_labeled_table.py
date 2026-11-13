import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q117"

from solutions.q117_labeled_table import labeled_table


class TestQ117(unittest.TestCase):
    def test_labeled_table(self):
        paths = [
            str(ROOT / 'fixtures' / 'dogs.png'),
            str(ROOT / 'fixtures' / 'images' / 'shapes.png'),
        ]
        labels = ['dog', 'shape']
        actual = labeled_table(paths, labels)
        self.assertEqual(list(actual.columns), ["path", "label", "width", "height"])
        self.assertEqual(actual['path'].tolist(), paths)
        self.assertEqual(actual['label'].tolist(), labels)
        self.assertEqual(actual['width'].tolist(), [48, 80])
        self.assertEqual(actual['height'].tolist(), [48, 80])


    def test_case_b(self):
        import cv2
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'pic.png')
        cv2.imwrite(path, np.zeros((14, 19, 3), np.uint8))
        actual = labeled_table([path], ["cat"])
        self.assertEqual(list(actual.columns), ["path", "label", "width", "height"])
        self.assertEqual(actual["label"].tolist(), ["cat"])
        self.assertEqual(actual["width"].tolist(), [19])
        self.assertEqual(actual["height"].tolist(), [14])

    def test_case_c(self):
        import cv2
        folder = tempfile.mkdtemp()
        first = str(Path(folder) / 'a.png')
        second = str(Path(folder) / 'b.png')
        cv2.imwrite(first, np.zeros((8, 12, 3), np.uint8))
        cv2.imwrite(second, np.zeros((22, 7, 3), np.uint8))
        actual = labeled_table([first, second], ["one", "two"])
        self.assertEqual(actual["label"].tolist(), ["one", "two"])
        self.assertEqual(actual["width"].tolist(), [12, 7])
        self.assertEqual(actual["height"].tolist(), [8, 22])

if __name__ == "__main__":
    unittest.main()
