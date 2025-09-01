# app/routers/tests_history_router.py
from __future__ import annotations

from typing import Any, Dict, List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse

from app.utils.mongo import db
from app.routers.auth_router import get_current_user
from app.services.pdf_report import build_test_result_pdf

router = APIRouter(prefix="/tests/history", tags=["tests-history"])


def _oid(val: str) -> ObjectId:
    try:
        return ObjectId(val)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


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
async def list_attempts(candidate_id: str = Path(...), current_user: Any = Depends(get_current_user)):
    """
    Return all test attempts for this candidate (newest first) with a PDF link.

    Expects submissions stored in `test_submissions` (see tests_router usage of
    evaluate_test details and score). Results shape is friendly for the History tab.
    """
    candidate = await _get_candidate_guarded(candidate_id, current_user)

    # Filter by candidate + owner guard if the collection stores ownerUserId
    owner_id = str(getattr(current_user, "id", None))
    query = {"candidateId": candidate_id}
    # add owner filter if present on docs (won't break if absent)
    attempts: List[Dict[str, Any]] = []
    cursor = db.test_submissions.find(query).sort("submittedAt", -1)
    async for doc in cursor:
        attempt_id = str(doc.get("_id"))
        score = doc.get("total_score") or doc.get("score")
        attempts.append(
            {
                "id": attempt_id,
                "submittedAt": doc.get("submittedAt"),
                "score": score,
                "pdfUrl": f"/tests/history/{candidate_id}/{attempt_id}/report.pdf",
            }
        )
    return {"candidateId": candidate_id, "attempts": attempts}


@router.get("/{candidate_id}/{attempt_id}/report.pdf")
async def attempt_report_pdf(
    candidate_id: str = Path(...),
    attempt_id: str = Path(...),
    current_user: Any = Depends(get_current_user),
):
    """
    Generate and stream a PDF report for a specific attempt (opens in browser).
    """
    candidate = await _get_candidate_guarded(candidate_id, current_user)
    attempt = await db.test_submissions.find_one({"_id": _oid(attempt_id), "candidateId": candidate_id})
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
