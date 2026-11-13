"""Read the generated problem manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "problems" / "bank"
MANIFEST = ROOT / "problems" / "manifest.json"


class Problem(TypedDict):
    id: str
    number: int
    title: str
    prompt: str
    topic: str
    type: str
    starter: str
    test_module: str
    test_class: str
    vanilla_test: str
    solution_file: str
    test_count: int
    timeout: int


def load_problems() -> list[Problem]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def by_id() -> dict[str, Problem]:
    return {problem["id"]: problem for problem in load_problems()}
