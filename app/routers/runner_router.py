# backend/routes/runner_router.py
"""
Runner API endpoints.

Exposes:
- GET   /runner/health         -> quick health check
- GET   /runner/languages      -> list languages supported by code_runner
- POST  /runner/run            -> execute a single submission against tests
- POST  /runner/batch          -> (optional) run multiple submissions sequentially

This router delegates execution to services.code_runner.run_submission(),
which handles compilation/execution, limits, and result shaping.

Wire it up in your FastAPI app:
    from fastapi import FastAPI
    from routes.runner_router import router as runner_router

    app = FastAPI()
    app.include_router(runner_router)

Security notes:
- This is a local runner not a hardened sandbox. For untrusted traffic, put the
  runner behind auth/rate-limits and consider container isolation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator

try:
    # Local import (your project layout): services/code_runner.py
    from services.code_runner import run_submission, LANGS
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import services.code_runner. "
        "Ensure backend/services/code_runner.py exists."
    ) from e


router = APIRouter(prefix="/runner", tags=["runner"])

# ---------------- Config ----------------
ALLOWED_LANGS = list(LANGS.keys())
MAX_SOURCE_BYTES = 200_000        # ~200 KB
MAX_TESTS = 25
DEFAULT_TIME_LIMIT = 2            # seconds
DEFAULT_MEMORY_LIMIT_MB = 128


# ---------------- Pydantic models ----------------
class TestCaseModel(BaseModel):
    name: str = Field(default="test")
    input: str = Field(default="")
    args: List[str] = Field(default_factory=list)
    expected_stdout: Optional[str] = Field(default=None)
    match: str = Field(default="exact", description="exact | contains | regex")
    timeout: Optional[int] = Field(default=None, ge=1, le=30, description="Per-test timeout (seconds)")

    @validator("match")
    def _match_mode(cls, v: str) -> str:
        mode = (v or "exact").lower()
        if mode not in {"exact", "contains", "regex"}:
            raise ValueError("match must be one of: exact, contains, regex")
        return mode


class SubmissionModel(BaseModel):
    language: str = Field(..., description=f"One of: {', '.join(ALLOWED_LANGS)}")
    source: str
    tests: List[TestCaseModel] = Field(default_factory=list)
    time_limit_sec: int = Field(default=DEFAULT_TIME_LIMIT, ge=1, le=30)
    memory_limit_mb: int = Field(default=DEFAULT_MEMORY_LIMIT_MB, ge=32, le=2048)

    @validator("language")
    def _lang_ok(cls, v: str) -> str:
        v = (v or "").lower().strip()
        if v not in ALLOWED_LANGS:
            raise ValueError(f"Unsupported language. Allowed: {', '.join(ALLOWED_LANGS)}")
        return v

    @validator("source")
    def _source_size(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source cannot be empty")
        if len(v.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ValueError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
        return v

    @validator("tests")
    def _tests_limit(cls, v: List[TestCaseModel]) -> List[TestCaseModel]:
        if len(v) > MAX_TESTS:
            raise ValueError(f"Too many tests (max {MAX_TESTS})")
        return v


class BatchModel(BaseModel):
    submissions: List[SubmissionModel] = Field(default_factory=list)


# ---------------- Routes ----------------
@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/languages")
async def list_languages() -> Dict[str, Dict[str, str]]:
    """
    Returns the languages that the underlying code_runner supports.
    """
    # Expose a minimal public view of the LANGS config
    return {
        "languages": {
            key: {"extension": spec.ext}
            for key, spec in LANGS.items()
        }
    }


@router.post("/run")
async def run(submission: SubmissionModel):
    """
    Execute a single submission against its test cases.
    """
    payload = submission.dict()
    try:
        # run_submission is CPU/subprocess-bound; keep event loop free.
        result = await run_in_threadpool(run_submission, payload)
        return result
    except Exception as e:
        # Normalize to 400 (client) for validation-ish issues; otherwise 500.
        msg = str(e)
        status = 400 if "Unsupported language" in msg or "must" in msg else 500
        raise HTTPException(status_code=status, detail=msg) from e


@router.post("/batch")
async def run_batch(batch: BatchModel):
    """
    Execute multiple submissions sequentially.
    Useful for regrading or test builds.
    """
    results = []
    for sub in batch.submissions:
        payload = sub.dict()
        try:
            res = await run_in_threadpool(run_submission, payload)
        except Exception as e:  # capture error per item
            res = {"ok": False, "error": str(e), "language": sub.language}
        results.append(res)
    passed = sum(1 for r in results if r.get("ok"))
    return {"ok": passed == len(results), "summary": {"passed": passed, "total": len(results)}, "results": results}
