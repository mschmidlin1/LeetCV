import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q057"

from solutions.q057_draw_boxes import draw_boxes


class TestQ057(unittest.TestCase):
    def test_draw_boxes(self):
        image = np.load(CASE / 'image.npy')
        boxes = np.load(CASE / 'boxes.npy')
        actual = draw_boxes(image, boxes)
        expected = np.load(CASE / 'expected.npy')
        np.testing.assert_array_equal(actual, expected)


    def test_case_b(self):
        image = np.load(CASE / 'case_b_image.npy')
        boxes = np.load(CASE / 'case_b_boxes.npy')
        actual = draw_boxes(image, boxes)
        expected = np.load(CASE / 'case_b_expected.npy')
        np.testing.assert_array_equal(actual, expected)

    def test_case_c(self):
        image = np.load(CASE / 'case_c_image.npy')
        boxes = np.load(CASE / 'case_c_boxes.npy')
        actual = draw_boxes(image, boxes)
        expected = np.load(CASE / 'case_c_expected.npy')
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
