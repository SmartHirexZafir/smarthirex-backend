# app/routers/tests_history_router.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse

from app.utils.mongo import db
from app.routers.auth_router import get_current_user
from app.services.pdf_report import build_test_result_pdf

router = APIRouter(prefix="/tests/history", tags=["tests-history"])


def _maybe_oid(val: str) -> Optional[ObjectId]:
  """
  Try to coerce to ObjectId; return None if not valid.
  We do NOT raise because our submissions _id may be UUID hex strings.
  """
  try:
    return ObjectId(val)
  except Exception:
    return None


def to_pct_number(v: Any) -> Optional[float]:
  """
  Normalize a possibly-ratio/percent-string score to a 0..100 float.
  Accepts:
    - 0..1 ratios (multiplied by 100)
    - plain numbers (clamped 0..100)
    - strings like '85' or '85%' or '0.85'
  Returns None if it cannot be parsed.
  """
  if v is None:
    return None
  if isinstance(v, (int, float)):
    n = float(v)
    return max(0.0, min(100.0, n * 100.0 if n <= 1.0 else n))
  if isinstance(v, str):
    s = v.strip().replace("%", "")
    try:
      n = float(s)
      return max(0.0, min(100.0, n * 100.0 if n <= 1.0 else n))
    except ValueError:
      return None
  return None


async def _get_candidate_guarded(candidate_id: str, current_user: Any) -> Dict[str, Any]:
  """
  Enforce ownership, mirroring candidate_router behavior (ownerUserId scope).
  """
  owner_id = str(getattr(current_user, "id", None))
  cand = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
  if not cand:
    raise HTTPException(status_code=404, detail="Candidate not found")
  return cand


@router.get("/{candidate_id}")
async def list_attempts(
  candidate_id: str = Path(...),
  current_user: Any = Depends(get_current_user),
):
  """
  Return all test attempts for this candidate (newest first) with a PDF link.

  Compatible with both SMART and CUSTOM tests:
   - Includes `type` ("smart" | "custom")
   - Includes `needs_marking` (True for custom tests awaiting manual grading)
   - Uses stored `score` (0.0 for ungraded custom)
  """
  await _get_candidate_guarded(candidate_id, current_user)

  query = {"candidateId": candidate_id}
  attempts: List[Dict[str, Any]] = []
  cursor = db.test_submissions.find(query).sort("submittedAt", -1)
  async for doc in cursor:
    attempt_id = str(doc.get("_id"))
    # ✅ Normalize to numeric 0..100 for frontend consistency (no percent strings)
    raw_score = doc.get("score") if doc.get("score") is not None else doc.get("total_score")
    score = to_pct_number(raw_score)
    
    # ✅ Return COMPLETE test data for History tab
    attempt_data = {
      "id": attempt_id,
      "submittedAt": doc.get("submittedAt"),
      "score": score,
      "type": doc.get("type", "smart"),
      "testType": doc.get("testType", doc.get("type", "smart")),  # Explicit test type
      "needs_marking": bool(doc.get("needs_marking", False)),
      "pdfUrl": f"/tests/history/{candidate_id}/{attempt_id}/report.pdf",
      # ✅ Complete test data
      "questions": doc.get("questions", []),  # All questions from the test
      "answers": doc.get("answers", []),  # Candidate's submitted answers
      "details": doc.get("details", []),  # Complete evaluation details (question, answer, is_correct, explanation, etc.)
      "custom": doc.get("custom"),  # Custom test metadata (title, etc.)
      "questionCount": doc.get("questionCount", len(doc.get("questions", []))),
    }
    attempts.append(attempt_data)
  return {"candidateId": candidate_id, "attempts": attempts}


@router.get("/{candidate_id}/{attempt_id}/report.pdf")
async def attempt_report_pdf(
  candidate_id: str = Path(...),
  attempt_id: str = Path(...),
  current_user: Any = Depends(get_current_user),
):
  """
  Generate and stream a PDF report for a specific attempt (opens in browser).

  Works with either UUID-string _id or ObjectId _id.
  """
  candidate = await _get_candidate_guarded(candidate_id, current_user)

  # Support both UUID hex string ids and legacy ObjectId
  maybe_oid = _maybe_oid(attempt_id)
  attempt_query = {
    "$and": [
      {"candidateId": candidate_id},
      {"$or": [{"_id": attempt_id}, *([{"_id": maybe_oid}] if maybe_oid else [])]},
    ]
  }
  attempt = await db.test_submissions.find_one(attempt_query)
  if not attempt:
    # Fallback: direct string match (in case $or path failed due to None handling)
    attempt = await db.test_submissions.find_one({"_id": attempt_id, "candidateId": candidate_id})
    if not attempt:
      raise HTTPException(status_code=404, detail="Attempt not found")

  pdf_bytes = await build_test_result_pdf(candidate, attempt)
  filename = f"assessment_{candidate.get('name','candidate')}_{attempt_id}.pdf".replace(" ", "_")
  return StreamingResponse(
    content=BytesIterator(pdf_bytes),
    media_type="application/pdf",
    headers={"Content-Disposition": f'inline; filename="{filename}"'},
  )


class BytesIterator:
  """
  FastAPI-compatible iterator for in-memory bytes.
  """
  def __init__(self, data: bytes, chunk_size: int = 64 * 1024):
    self._data = data
    self._n = len(data)
    self._i = 0
    self._chunk = chunk_size

  def __iter__(self):
    return self

  def __next__(self) -> bytes:
    if self._i >= self._n:
      raise StopIteration
    j = min(self._i + self._chunk, self._n)
    chunk = self._data[self._i:j]
    self._i = j
    return chunk
