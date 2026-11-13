"""Run one problem's unit tests in a subprocess. Prints one JSON object."""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import io
import json
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

CALLS: list[dict] = []
_IMPORT = re.compile(r"^from (solutions\.\w+) import (.+)$", re.M)


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
        CALLS.clear()
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
                "stdout": self._stdout.getvalue()[-8000:],
                "stderr": self._stderr.getvalue()[-8000:],
                "returns": list(CALLS),
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


def summarize(value: object) -> str:
    if isinstance(value, np.ndarray):
        text = f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
        if value.size <= 24:
            text += "\n" + np.array2string(value, threshold=24)
        return text
    if type(value).__name__ == "DataFrame":
        return f"DataFrame shape={value.shape} columns={list(value.columns)}"
    if isinstance(value, (list, tuple)):
        shown = list(value[:8])
        parts = [summarize(item) for item in shown]
        if len(value) > 8:
            parts.append(f"… {len(value)} items")
        body = ", ".join(parts)
        return f"[{body}]" if isinstance(value, list) else f"({body})"
    text = repr(value)
    if len(text) > 500:
        return text[:500] + "…"
    return text


def _wrap(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:
            CALLS.append({"function": fn.__name__, "raised": f"{type(exc).__name__}: {exc}"})
            raise
        CALLS.append({"function": fn.__name__, "value": summarize(value)})
        return value

    wrapper._leetcv_wrapped = True
    return wrapper


def install_wrappers(source: str) -> None:
    """Wrap solution functions before the test module binds them with import."""
    for match in _IMPORT.finditer(source):
        module = importlib.import_module(match.group(1))
        for raw in match.group(2).split(","):
            name = raw.strip().split(" as ")[0].strip()
            if not name:
                continue
            obj = getattr(module, name)
            if inspect.isfunction(obj) and not getattr(obj, "_leetcv_wrapped", False):
                setattr(module, name, _wrap(obj))


def load_suite(root: Path, test_module: str, qual: str) -> unittest.TestSuite:
    stem = test_module.split(".")[-1]
    path = root / "tests" / f"{stem}.py"
    install_wrappers(path.read_text(encoding="utf-8"))
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
    captured_out, captured_err = io.StringIO(), io.StringIO()
    sys.stdout = captured_out
    sys.stderr = captured_err
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
                    "stderr": "",
                    "returns": [],
                    "duration_ms": 0,
                }
            )
        payload = {
            "tests": result.rows,
            "stdout": captured_out.getvalue()[-8000:],
            "stderr": captured_err.getvalue()[-8000:],
        }
    except Exception:
        payload = {
            "tests": [
                {
                    "name": qual,
                    "passed": False,
                    "traceback": traceback.format_exc()[-12000:],
                    "stdout": captured_out.getvalue()[-8000:],
                    "stderr": captured_err.getvalue()[-8000:],
                    "returns": [],
                    "duration_ms": 0,
                }
            ],
            "stdout": captured_out.getvalue()[-8000:],
            "stderr": captured_err.getvalue()[-8000:],
        }
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
    json.dump(payload, real_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
