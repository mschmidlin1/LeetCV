"""Global progress stored in one SQLite file."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def data_dir() -> Path:
    configured = os.environ.get("LEETCV_DATA")
    path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect():
    connection = sqlite3.connect(data_dir() / "leetcv.db")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            problem_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            code TEXT,
            essay TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def all_progress() -> dict[str, sqlite3.Row]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM progress").fetchall()
    return {row["problem_id"]: row for row in rows}


def get_progress(problem_id: str) -> sqlite3.Row | None:
    with connect() as connection:
        return connection.execute(
            "SELECT * FROM progress WHERE problem_id = ?", (problem_id,)
        ).fetchone()


def save_progress(
    problem_id: str,
    *,
    status: str | None = None,
    code: str | None = None,
    essay: str | None = None,
) -> str:
    """Store code or an essay. A solved problem stays solved."""
    current = get_progress(problem_id)
    current_status = current["status"] if current else "todo"
    next_status = status or current_status
    if current_status == "solved":
        next_status = "solved"
    if next_status == "todo":
        next_status = "attempted"
    next_code = code if code is not None else (current["code"] if current else None)
    next_essay = essay if essay is not None else (current["essay"] if current else None)
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO progress (problem_id, status, code, essay, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(problem_id) DO UPDATE SET
                status = excluded.status,
                code = excluded.code,
                essay = excluded.essay,
                updated_at = excluded.updated_at
            """,
            (problem_id, next_status, next_code, next_essay, _now()),
        )
    return next_status
