"""Copy a problem into a temp directory and grade the submitted file."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.problems import BANK, Problem

RUNNER = Path(__file__).resolve().parent / "runner.py"


def grade(problem: Problem, code: str, *, submit: bool) -> dict:
    if problem["type"] != "code":
        raise ValueError("only code problems can be graded")
    qual = problem["test_class"]
    if not submit:
        qual = f"{qual}.{problem['vanilla_test']}"
    temp = Path(tempfile.mkdtemp(prefix="leetcv-"))
    try:
        shutil.copytree(BANK / "tests", temp / "tests")
        shutil.copytree(BANK / "solutions", temp / "solutions")
        shutil.copytree(BANK / "fixtures", temp / "fixtures")
        (temp / problem["solution_file"]).write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), str(temp), problem["test_module"], qual],
                cwd=temp,
                capture_output=True,
                text=True,
                timeout=int(problem["timeout"]),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "tests": [
                    {
                        "name": qual,
                        "passed": False,
                        "traceback": f"Timed out after {problem['timeout']}s.",
                        "stdout": "",
                        "stderr": "",
                        "returns": [],
                        "duration_ms": int(problem["timeout"]) * 1000,
                    }
                ],
                "stdout": "",
                "stderr": "",
            }
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            detail = (completed.stderr or completed.stdout or "The judge crashed.").strip()
            payload = {
                "tests": [
                    {
                        "name": qual,
                        "passed": False,
                        "traceback": detail[-12000:],
                        "stdout": "",
                        "stderr": "",
                        "returns": [],
                        "duration_ms": 0,
                    }
                ],
                "stdout": "",
                "stderr": "",
            }
        tests = payload.get("tests") or []
        return {
            "passed": bool(tests) and all(item.get("passed") for item in tests),
            "tests": tests,
            "stdout": payload.get("stdout") or "",
            "stderr": payload.get("stderr") or "",
        }
    finally:
        shutil.rmtree(temp, ignore_errors=True)
