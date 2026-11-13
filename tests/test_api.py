import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.app import app

GRAY = """from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LEETCV_DATA"] = self.tmp.name
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_catalog_counts_and_topics(self) -> None:
        response = self.client.get("/api/problems")
        self.assertEqual(response.status_code, 200)
        problems = response.json()
        self.assertEqual(len(problems), 120)
        self.assertEqual(sum(item["type"] == "code" for item in problems), 106)
        self.assertEqual(sum(item["type"] == "critical_thinking" for item in problems), 14)
        self.assertTrue(all(item["topic"] and item["status"] == "todo" for item in problems))
        gray = next(item for item in problems if item["id"] == "q005")
        self.assertEqual(gray["test_count"], 3)
        self.assertNotIn("q015", {item["id"] for item in problems})

    def test_code_run_does_not_solve_and_submit_does(self) -> None:
        detail = self.client.get("/api/problems/q005").json()
        self.assertIn("NotImplementedError", detail["starter"])
        failed = self.client.post("/api/problems/q005/run", json={"code": detail["starter"]})
        self.assertEqual(failed.status_code, 200)
        self.assertFalse(failed.json()["passed"])
        self.assertEqual(failed.json()["status"], "attempted")
        self.assertEqual(len(failed.json()["tests"]), 1)

        sample = self.client.post("/api/problems/q005/run", json={"code": GRAY})
        self.assertTrue(sample.json()["passed"])
        self.assertEqual(sample.json()["status"], "attempted")

        accepted = self.client.post("/api/problems/q005/submit", json={"code": GRAY})
        self.assertTrue(accepted.json()["passed"])
        self.assertEqual(accepted.json()["status"], "solved")
        self.assertEqual(len(accepted.json()["tests"]), 3)

        later = self.client.post("/api/problems/q005/submit", json={"code": detail["starter"]})
        self.assertFalse(later.json()["passed"])
        self.assertEqual(later.json()["status"], "solved")

    def test_critical_thinking_save_and_solve(self) -> None:
        blocked = self.client.post("/api/problems/q003/solve")
        self.assertEqual(blocked.status_code, 400)
        saved = self.client.put("/api/problems/q003/essay", json={"essay": "Gray is one channel."})
        self.assertEqual(saved.json()["status"], "attempted")
        solved = self.client.post("/api/problems/q003/solve")
        self.assertEqual(solved.json()["status"], "solved")
        edited = self.client.put(
            "/api/problems/q003/essay", json={"essay": "Gray is one channel. Color is three."}
        )
        self.assertEqual(edited.json()["status"], "solved")
        detail = self.client.get("/api/problems/q003").json()
        self.assertEqual(detail["essay"], "Gray is one channel. Color is three.")
        self.assertEqual(detail["type"], "critical_thinking")


if __name__ == "__main__":
    unittest.main()
