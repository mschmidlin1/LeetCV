import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "fixtures" / "cases" / "q092"

from solutions.q092_classify_patch import classify_patch


class TestQ092(unittest.TestCase):
    def test_classify_patch(self):
        features = np.load(CASE / 'features.npy')
        labels = np.load(CASE / 'labels.npy')
        sample = np.load(CASE / 'sample.npy')
        actual = classify_patch(features, labels, sample)
        expected = json.loads((CASE / 'expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


    def test_case_b(self):
        features = np.load(CASE / 'case_b_features.npy')
        labels = np.load(CASE / 'case_b_labels.npy')
        sample = np.load(CASE / 'case_b_sample.npy')
        actual = classify_patch(features, labels, sample)
        expected = json.loads((CASE / 'case_b_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))

    def test_case_c(self):
        features = np.load(CASE / 'case_c_features.npy')
        labels = np.load(CASE / 'case_c_labels.npy')
        sample = np.load(CASE / 'case_c_sample.npy')
        actual = classify_patch(features, labels, sample)
        expected = json.loads((CASE / 'case_c_expected.json').read_text(encoding='utf-8'))
        self.assertEqual(actual, expected if not isinstance(expected, list) else tuple(expected))


if __name__ == "__main__":
    unittest.main()
