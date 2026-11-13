import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q004"

from solutions.q004_save_image import save_image, save_jpeg


class TestQ004(unittest.TestCase):
    def test_save_image(self):
        image = np.load(CASE / 'image.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.png')
        save_image(image, path)
        self.assertEqual(Path(path).read_bytes(), (CASE / 'expected.bin').read_bytes())
        jpg_folder = tempfile.mkdtemp()
        jpg = str(Path(jpg_folder) / 'out.jpg')
        save_jpeg(image, jpg)
        np.testing.assert_array_equal(cv2.imread(jpg), np.load(CASE / 'expected_jpeg.npy'))

    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.png')
        save_image(image, path)
        np.testing.assert_array_equal(cv2.imread(path), image)
        jpg_folder = tempfile.mkdtemp()
        jpg = str(Path(jpg_folder) / 'out.jpg')
        save_jpeg(image, jpg)
        np.testing.assert_array_equal(cv2.imread(jpg), np.load(CASE / 'case_b_expected_jpeg.npy'))

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        folder = tempfile.mkdtemp()
        path = str(Path(folder) / 'out.png')
        save_image(image, path)
        np.testing.assert_array_equal(cv2.imread(path), image)
        jpg_folder = tempfile.mkdtemp()
        jpg = str(Path(jpg_folder) / 'out.jpg')
        save_jpeg(image, jpg)
        np.testing.assert_array_equal(cv2.imread(jpg), np.load(CASE / 'case_c_expected_jpeg.npy'))


if __name__ == "__main__":
    unittest.main()
