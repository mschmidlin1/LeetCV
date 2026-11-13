"""Run the original bank tests against the oracle implementations."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "problems" / "bank"
TOOLS = ROOT / "tools"


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        shutil.copytree(BANK / "fixtures", temp / "fixtures")
        shutil.copytree(BANK / "tests", temp / "tests")
        solutions = temp / "solutions"
        solutions.mkdir()
        (solutions / "__init__.py").write_text("", encoding="utf-8")
        for source in sorted((BANK / "solutions").glob("q*.py")):
            function = source.stem.split("_", 1)[1]
            wrapper = (
                "import sys\n"
                f"sys.path.insert(0, {str(TOOLS)!r})\n"
                f"from oracles.impl import {function}, OpenCVError\n"
            )
            (solutions / source.name).write_text(wrapper, encoding="utf-8")
        sys.path.insert(0, str(temp))
        loader = unittest.defaultTestLoader
        suite = loader.discover(str(temp / "tests"), pattern=f"test_{only}*.py" if only else "test_q*.py")
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
        print(f"\nran={result.testsRun} failed={len(result.failures)} errors={len(result.errors)}")
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
