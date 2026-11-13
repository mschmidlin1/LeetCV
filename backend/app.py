"""LeetCV API and, when the frontend has been built, the single-page app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.db import all_progress, get_progress, save_progress
from backend.judge import grade
from backend.problems import by_id

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
app = FastAPI(title="LeetCV")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeBody(BaseModel):
    code: str = Field(min_length=1, max_length=200_000)


class EssayBody(BaseModel):
    essay: str = Field(max_length=100_000)


def _status(problem_id: str) -> str:
    row = get_progress(problem_id)
    return row["status"] if row else "todo"


def _summary(problem: dict, status: str) -> dict:
    return {
        "id": problem["id"],
        "number": problem["number"],
        "title": problem["title"],
        "topic": problem["topic"],
        "type": problem["type"],
        "status": status,
        "test_count": problem["test_count"],
    }


def _detail(problem: dict) -> dict:
    row = get_progress(problem["id"])
    status = row["status"] if row else "todo"
    saved_code = row["code"] if row and row["code"] is not None else None
    essay = row["essay"] if row and row["essay"] is not None else ""
    return {
        **_summary(problem, status),
        "prompt": problem["prompt"],
        "starter": problem["starter"],
        "code": saved_code if saved_code is not None else problem["starter"],
        "essay": essay or "",
    }


def _require(problem_id: str) -> dict:
    problem = by_id().get(problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problem


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/problems")
def list_problems() -> list[dict]:
    progress = all_progress()
    listed = []
    for problem in by_id().values():
        row = progress.get(problem["id"])
        listed.append(_summary(problem, row["status"] if row else "todo"))
    return listed


@app.get("/api/problems/{problem_id}")
def get_problem(problem_id: str) -> dict:
    return _detail(_require(problem_id))


@app.put("/api/problems/{problem_id}/code")
def save_code(problem_id: str, body: CodeBody) -> dict:
    problem = _require(problem_id)
    if problem["type"] != "code":
        raise HTTPException(status_code=400, detail="This problem has no editor")
    status = save_progress(problem_id, status="attempted", code=body.code)
    return {"status": status}


@app.put("/api/problems/{problem_id}/essay")
def save_essay(problem_id: str, body: EssayBody) -> dict:
    problem = _require(problem_id)
    if problem["type"] != "critical_thinking":
        raise HTTPException(status_code=400, detail="This problem has no written answer")
    status = save_progress(problem_id, status="attempted", essay=body.essay)
    return {"status": status}


@app.post("/api/problems/{problem_id}/solve")
def mark_solved(problem_id: str) -> dict:
    problem = _require(problem_id)
    if problem["type"] != "critical_thinking":
        raise HTTPException(status_code=400, detail="Code problems are solved by Submit")
    row = get_progress(problem_id)
    essay = (row["essay"] if row and row["essay"] else "").strip()
    if not essay:
        raise HTTPException(status_code=400, detail="Save an answer before marking it solved")
    status = save_progress(problem_id, status="solved")
    return {"status": status}


def _grade(problem_id: str, code: str, *, submit: bool) -> dict:
    problem = _require(problem_id)
    if problem["type"] != "code":
        raise HTTPException(status_code=400, detail="This problem is not graded")
    result = grade(problem, code, submit=submit)
    if submit and result["passed"]:
        status = save_progress(problem_id, status="solved", code=code)
    else:
        status = save_progress(problem_id, status="attempted", code=code)
    return {**result, "status": status}


@app.post("/api/problems/{problem_id}/run")
def run_problem(problem_id: str, body: CodeBody) -> dict:
    return _grade(problem_id, body.code, submit=False)


@app.post("/api/problems/{problem_id}/submit")
def submit_problem(problem_id: str, body: CodeBody) -> dict:
    return _grade(problem_id, body.code, submit=True)


@app.get("/")
def index():
    page = DIST / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"status": "ok"})


@app.get("/{full_path:path}")
def frontend(full_path: str):
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not found")
    root = DIST.resolve()
    target = (DIST / full_path).resolve()
    if root in target.parents and target.is_file():
        return FileResponse(target)
    page = DIST / "index.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="Not found")
