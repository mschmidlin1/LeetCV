"""Build problems/manifest.json from the bank and the interview question list."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "problems" / "bank"
QUESTIONS = Path(r"C:\Users\mschm\source\Computer-Vision-Interview-Preparation\interview question.md")

CONCEPTS = {3, 31, 42, 56, 60, 64, 66, 68, 79, 89, 105, 108, 110, 134}
EXCLUDED = {15, 33, 47, 54, 63, 65, 75, 77, 78, 94, 95, 96, 97, 102, 106, 107, 115, 116, 128, 133, 138}
SLOW = {28, 40, 43, 51, 52, 53, 55, 61, 62, 67, 70, 71, 72, 76, 80, 81, 83, 84, 85, 86, 87, 90, 103}

TOPICS = {
    1: "image basics",
    2: "image basics",
    3: "image basics",
    4: "image basics",
    5: "color",
    6: "color",
    7: "transforms",
    8: "image basics",
    9: "image basics",
    10: "image basics",
    11: "transforms",
    12: "transforms",
    13: "filtering",
    14: "filtering",
    16: "filtering",
    17: "filtering",
    18: "filtering",
    19: "color",
    20: "filtering",
    21: "filtering",
    22: "filtering",
    23: "filtering",
    24: "filtering",
    25: "features",
    26: "features",
    27: "features",
    28: "features",
    29: "features",
    30: "features",
    31: "features",
    32: "features",
    34: "transforms",
    35: "transforms",
    36: "transforms",
    37: "transforms",
    38: "transforms",
    39: "transforms",
    40: "transforms",
    41: "video",
    42: "video",
    43: "video",
    44: "video",
    45: "video",
    46: "video",
    48: "video",
    49: "video",
    50: "video",
    51: "video",
    52: "video",
    53: "detection",
    55: "detection",
    56: "detection",
    57: "detection",
    58: "detection",
    59: "detection",
    60: "detection",
    61: "detection",
    62: "detection",
    64: "detection",
    66: "learning",
    67: "video",
    68: "video",
    69: "segmentation",
    70: "segmentation",
    71: "segmentation",
    72: "segmentation",
    73: "segmentation",
    74: "segmentation",
    76: "segmentation",
    79: "calibration",
    80: "calibration",
    81: "calibration",
    82: "calibration",
    83: "calibration",
    84: "calibration",
    85: "calibration",
    86: "calibration",
    87: "calibration",
    88: "learning",
    89: "learning",
    90: "learning",
    91: "learning",
    92: "learning",
    93: "learning",
    98: "systems",
    99: "systems",
    100: "systems",
    101: "systems",
    103: "systems",
    104: "systems",
    105: "systems",
    108: "systems",
    109: "systems",
    110: "systems",
    111: "systems",
    112: "systems",
    113: "learning",
    114: "learning",
    117: "systems",
    118: "systems",
    119: "systems",
    120: "systems",
    121: "systems",
    122: "systems",
    123: "systems",
    124: "systems",
    125: "systems",
    126: "systems",
    127: "applications",
    129: "applications",
    130: "applications",
    131: "applications",
    132: "applications",
    134: "applications",
    135: "applications",
    136: "applications",
    137: "applications",
    139: "applications",
    140: "applications",
    141: "applications",
}


def prompts() -> dict[int, str]:
    found: dict[int, str] = {}
    current: int | None = None
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\d+)\.\s+(.*)", line)
        if match:
            current = int(match.group(1))
            found[current] = match.group(2).strip()
        elif current is not None and line.strip() and not line.startswith("#"):
            found[current] += " " + line.strip()
    return found


def title_of(body: str) -> str:
    head = body.split(" Implement `", 1)[0].strip()
    if head.endswith("."):
        head = head[:-1]
    return head


def test_meta(path: Path) -> tuple[str, str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        names = [
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test")
        ]
        vanilla = next(name for name in names if not name.startswith("test_case_"))
        return node.name, vanilla, len(names)
    raise RuntimeError(f"no test class in {path.name}")


def main() -> None:
    texts = prompts()
    problems = []
    for number in range(1, 142):
        if number in EXCLUDED:
            continue
        if number not in TOPICS:
            raise SystemExit(f"missing topic for {number}")
        body = texts[number]
        problem_id = f"q{number:03d}"
        entry = {
            "id": problem_id,
            "number": number,
            "title": title_of(body),
            "prompt": body,
            "topic": TOPICS[number],
            "timeout": 120 if number in SLOW else 30,
        }
        if number in CONCEPTS:
            entry.update(
                {
                    "type": "critical_thinking",
                    "starter": "",
                    "test_module": "",
                    "test_class": "",
                    "vanilla_test": "",
                    "solution_file": "",
                    "test_count": 0,
                }
            )
        else:
            solution = next((BANK / "solutions").glob(f"{problem_id}_*.py"))
            test = next((BANK / "tests").glob(f"test_{problem_id}_*.py"))
            test_class, vanilla, count = test_meta(test)
            entry.update(
                {
                    "type": "code",
                    "starter": solution.read_text(encoding="utf-8"),
                    "test_module": f"tests.{test.stem}",
                    "test_class": test_class,
                    "vanilla_test": vanilla,
                    "solution_file": f"solutions/{solution.name}",
                    "test_count": count,
                }
            )
        problems.append(entry)
    code = sum(1 for item in problems if item["type"] == "code")
    concepts = sum(1 for item in problems if item["type"] == "critical_thinking")
    if code != 106 or concepts != 14:
        raise SystemExit(f"unexpected counts code={code} concepts={concepts}")
    destination = ROOT / "problems" / "manifest.json"
    destination.write_text(json.dumps(problems, indent=2), encoding="utf-8")
    print(f"wrote {len(problems)} problems to {destination}")


if __name__ == "__main__":
    main()
