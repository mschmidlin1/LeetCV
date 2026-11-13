"""Run one problem's unit tests in a subprocess. Prints one JSON object."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import traceback
import unittest
from pathlib import Path


class Collect(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict] = []
        self._started = 0.0
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._saved: tuple[io.TextIOBase, io.TextIOBase] | None = None

    def startTest(self, test: unittest.TestCase) -> None:
        self._started = time.perf_counter()
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
        super().stopTest(test)

    def _row(self, test: unittest.TestCase, passed: bool, detail: str) -> None:
        duration = int((time.perf_counter() - self._started) * 1000)
        self.rows.append(
            {
                "name": getattr(test, "_testMethodName", str(test)),
                "passed": passed,
                "traceback": detail[-12000:],
                "stdout": (self._stdout.getvalue() + self._stderr.getvalue())[-8000:],
                "duration_ms": duration,
            }
        )

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._row(test, True, "")

    def addFailure(self, test: unittest.TestCase, err) -> None:
        super().addFailure(test, err)
        self._row(test, False, "".join(traceback.format_exception(*err)))

    def addError(self, test: unittest.TestCase, err) -> None:
        super().addError(test, err)
        self._row(test, False, "".join(traceback.format_exception(*err)))


def load_suite(root: Path, test_module: str, qual: str) -> unittest.TestSuite:
    stem = test_module.split(".")[-1]
    path = root / "tests" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    loader = unittest.defaultTestLoader
    if "." in qual:
        return loader.loadTestsFromName(qual, module)
    return loader.loadTestsFromTestCase(getattr(module, qual))


def main() -> int:
    root = Path(sys.argv[1])
    test_module = sys.argv[2]
    qual = sys.argv[3]
    sys.path.insert(0, str(root))
    real_out, real_err = sys.stdout, sys.stderr
    captured = io.StringIO()
    sys.stdout = captured
    sys.stderr = captured
    try:
        suite = load_suite(root, test_module, qual)
        result = Collect()
        suite.run(result)
        if result.testsRun == 0:
            result.rows.append(
                {
                    "name": qual,
                    "passed": False,
                    "traceback": "No tests ran.",
                    "stdout": "",
                    "duration_ms": 0,
                }
            )
        payload = {"tests": result.rows, "stdout": captured.getvalue()[-8000:]}
    except Exception:
        payload = {
            "tests": [
                {
                    "name": qual,
                    "passed": False,
                    "traceback": traceback.format_exc()[-12000:],
                    "stdout": captured.getvalue()[-8000:],
                    "duration_ms": 0,
                }
            ],
            "stdout": captured.getvalue()[-8000:],
        }
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
    json.dump(payload, real_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
