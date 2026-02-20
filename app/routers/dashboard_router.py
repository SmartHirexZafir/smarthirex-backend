# app/routers/dashboard_router.py
from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.utils.mongo import db  # AsyncIOMotorDatabase (Motor)
from app.routers.auth_router import get_current_user
from app.utils.datetime_serialization import serialize_utc_any

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

StatusBucket = TypedDict(
    "StatusBucket",
    {
        "count": int,
        "avg_total_score": Optional[float],
        "avg_test_score": Optional[float],
    },
    total=False,
)

CandidateOut = TypedDict(
    "CandidateOut",
    {
        "_id": str,
        "name": Optional[str],
        "email": Optional[str],
        "avatar": Optional[str],
        "job_role": Optional[str],
        "status": Optional[str],
        "score": Optional[float],         # match score
        "test_score": Optional[float],    # test %
        "total_score": Optional[float],   # avg(score, test_score)
        "rank": Optional[int],
        "updatedAt": Optional[str],
        "createdAt": Optional[str],
    },
    total=False,
)


def _email_of(doc: Dict[str, Any]) -> Optional[str]:
    return (doc.get("email") or (doc.get("resume") or {}).get("email") or None)


def _job_role_of(doc: Dict[str, Any]) -> str:
    return (
        doc.get("job_role")
        or doc.get("category")
        or doc.get("predicted_role")
        or (doc.get("resume") or {}).get("role")
        or "General"
    )


def _coerce_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _ensure_total_score(doc: Dict[str, Any]) -> float:
    """
    Return doc['total_score'] if present; otherwise compute avg(match, test).
    Does not persist (read-only).
    """
    if doc.get("total_score") is not None:
        try:
            return float(doc["total_score"])
        except Exception:
            pass
    match_score = _coerce_float(doc.get("score"))
    test_score = _coerce_float(doc.get("test_score", doc.get("testScore")))
    if match_score == 0.0 and test_score == 0.0:
        return 0.0
    return round((match_score + test_score) / 2.0, 1)


def _normalize_candidate(doc: Dict[str, Any]) -> CandidateOut:
    return {
        "_id": str(doc.get("_id")),
        "name": doc.get("name"),
        "email": _email_of(doc),
        "avatar": doc.get("avatar"),
        "job_role": _job_role_of(doc),
        "status": doc.get("status") or "new",
        "score": _coerce_float(doc.get("score")),
        "test_score": _coerce_float(doc.get("test_score", doc.get("testScore"))),
        "total_score": _ensure_total_score(doc),
        "rank": doc.get("rank"),
        # Normalize timestamps to ISO strings if present
        "updatedAt": (
            serialize_utc_any(doc.get("updatedAt")) or doc.get("updatedAt")
        ),
        "createdAt": (
            serialize_utc_any(doc.get("createdAt")) or doc.get("createdAt")
        ),
    }


async def _fetch_candidates_by_status(
    owner_id: str,
    statuses: List[str],
    limit: int,
    include_no_status: bool = False,
) -> List[CandidateOut]:
    """
    Fetch a list of candidates for given status buckets, sorted by total_score desc.
    """
    query: Dict[str, Any] = {"ownerUserId": owner_id}
    if include_no_status:
        query["$or"] = [{"status": {"$in": statuses}}, {"status": {"$exists": False}}]
    else:
        query["status"] = {"$in": statuses}

    projection = {
        "_id": 1,
        "name": 1,
        "email": 1,
        "resume.email": 1,
        "avatar": 1,
        "status": 1,
        "score": 1,           # match score
        "test_score": 1,      # snake_case
        "testScore": 1,       # camelCase (legacy)
        "rank": 1,
        "category": 1,
        "predicted_role": 1,
        "job_role": 1,
        "updatedAt": 1,
        "createdAt": 1,
        "total_score": 1,
    }

    items: List[CandidateOut] = []
    cursor = db.parsed_resumes.find(query, projection=projection).limit(limit)
    async for doc in cursor:
        items.append(_normalize_candidate(doc))

    # Sort by total_score desc, then match score desc
    items.sort(key=lambda d: (d.get("total_score") or 0, d.get("score") or 0), reverse=True)
    return items


async def _bucket_stats(items: List[CandidateOut]) -> StatusBucket:
    if not items:
        return {"count": 0, "avg_total_score": None, "avg_test_score": None}
    total_scores = [float(x.get("total_score") or 0) for x in items]
    test_scores = [float(x.get("test_score") or 0) for x in items]
    avg_total = round(sum(total_scores) / len(total_scores), 2) if total_scores else None
    avg_test = round(sum(test_scores) / len(test_scores), 2) if test_scores else None
    return {"count": len(items), "avg_total_score": avg_total, "avg_test_score": avg_test}


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard/summary
#   High-level KPIs + counts per bucket. Read-only, scoped to owner.
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/summary")
async def dashboard_summary(
    limit_per_bucket: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
):
    """
    Returns KPIs and per-bucket stats for the Dashboard.

    Buckets (initial):
      - accepted: status = "accepted"
      - rejected: status = "rejected"
      - shortlisted: status = "shortlisted"
      - in_review: status in ["new", "interviewed"] or missing
    """
    owner_id = str(getattr(current_user, "id", None))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    accepted = await _fetch_candidates_by_status(owner_id, ["accepted"], limit_per_bucket)
    rejected = await _fetch_candidates_by_status(owner_id, ["rejected"], limit_per_bucket)
    shortlisted = await _fetch_candidates_by_status(owner_id, ["shortlisted"], limit_per_bucket)
    in_review = await _fetch_candidates_by_status(
        owner_id, ["new", "interviewed"], limit_per_bucket, include_no_status=True
    )

    # KPIs
    total_candidates = len(accepted) + len(rejected) + len(shortlisted) + len(in_review)
    top_5 = sorted(
        accepted + shortlisted + in_review,
        key=lambda d: (d.get("total_score") or 0, d.get("score") or 0),
        reverse=True,
    )[:5]

    summary = {
        "counts": {
            "total": total_candidates,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "shortlisted": len(shortlisted),
            "in_review": len(in_review),
        },
        "stats": {
            "accepted": await _bucket_stats(accepted),
            "rejected": await _bucket_stats(rejected),
            "shortlisted": await _bucket_stats(shortlisted),
            "in_review": await _bucket_stats(in_review),
        },
        "leaderboard": top_5,
    }

    return JSONResponse(summary)


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard/lists
#   Full lists per bucket; supports pagination-like limiting.
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/lists")
async def dashboard_lists(
    limit_per_bucket: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
):
    """
    Returns normalized candidate lists for each Dashboard bucket.
    """
    owner_id = str(getattr(current_user, "id", None))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    accepted = await _fetch_candidates_by_status(owner_id, ["accepted"], limit_per_bucket)
    rejected = await _fetch_candidates_by_status(owner_id, ["rejected"], limit_per_bucket)
    shortlisted = await _fetch_candidates_by_status(owner_id, ["shortlisted"], limit_per_bucket)
    in_review = await _fetch_candidates_by_status(
        owner_id, ["new", "interviewed"], limit_per_bucket, include_no_status=True
    )

    return JSONResponse(
        {
            "accepted": accepted,
            "rejected": rejected,
            "shortlisted": shortlisted,
            "in_review": in_review,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# ✅ NEW (alias): GET /dashboard/overview
#   Backward-compatible alias that returns the same shape as /dashboard/lists.
#   This keeps older frontends working without code changes.
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/overview")
async def dashboard_overview(
    limit_per_bucket: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
) -> JSONResponse:
    # Delegate to the canonical /lists implementation
    return await dashboard_lists(limit_per_bucket=limit_per_bucket, current_user=current_user)  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard/leaderboard
#   Top N candidates across accepted/shortlisted/in_review by total_score.
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/leaderboard")
async def leaderboard(
    top_n: int = Query(10, ge=1, le=100),
    current_user: Any = Depends(get_current_user),
):
    owner_id = str(getattr(current_user, "id", None))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    buckets = []
    for statuses in (["accepted"], ["shortlisted"], ["new", "interviewed"]):
        items = await _fetch_candidates_by_status(owner_id, statuses, top_n * 2, include_no_status=(statuses != ["accepted"]))
        buckets.extend(items)

    buckets.sort(key=lambda d: (d.get("total_score") or 0, d.get("score") or 0), reverse=True)
    return {"leaderboard": buckets[:top_n]}


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard/export.csv
#   CSV export for Accepted and Rejected lists (common reporting need).
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/export.csv")
async def export_csv(
    include: str = Query("accepted,rejected", description="Comma-separated buckets: accepted,rejected,shortlisted,in_review"),
    limit_per_bucket: int = Query(1000, ge=1, le=5000),
    current_user: Any = Depends(get_current_user),
):
    owner_id = str(getattr(current_user, "id", None))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    buckets_requested = {b.strip().lower() for b in include.split(",") if b.strip()}
    allowed = {"accepted", "rejected", "shortlisted", "in_review"}
    buckets_requested = buckets_requested.intersection(allowed)
    if not buckets_requested:
        raise HTTPException(status_code=400, detail="No valid buckets requested")

    # fetch in the same order as requested
    bucket_fetch_map = {
        "accepted": (["accepted"], False),
        "rejected": (["rejected"], False),
        "shortlisted": (["shortlisted"], False),
        "in_review": (["new", "interviewed"], True),
    }

    rows: List[Dict[str, Any]] = []
    for b in include.split(","):
        key = b.strip().lower()
        if key not in buckets_requested:
            continue
        statuses, include_no_status = bucket_fetch_map[key]
        items = await _fetch_candidates_by_status(owner_id, statuses, limit_per_bucket, include_no_status=include_no_status)
        for x in items:
            rows.append(
                {
                    "bucket": key,
                    "id": x["_id"],
                    "name": x.get("name"),
                    "email": x.get("email"),
                    "job_role": x.get("job_role"),
                    "status": x.get("status"),
                    "match_score": x.get("score"),
                    "test_score": x.get("test_score"),
                    "total_score": x.get("total_score"),
                    "rank": x.get("rank"),
                    "updatedAt": x.get("updatedAt"),
                    "createdAt": x.get("createdAt"),
                }
            )

    # Build CSV in-memory
    headers = [
        "bucket",
        "id",
        "name",
        "email",
        "job_role",
        "status",
        "match_score",
        "test_score",
        "total_score",
        "rank",
        "updatedAt",
        "createdAt",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    buf.seek(0)

    filename = f"dashboard_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /dashboard/recent
#   Convenience endpoint: last N updated candidates (any status) for activity feed.
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/recent")
async def recent_activity(
    limit: int = Query(20, ge=1, le=200),
    current_user: Any = Depends(get_current_user),
):
    owner_id = str(getattr(current_user, "id", None))
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    projection = {
        "_id": 1,
        "name": 1,
        "email": 1,
        "resume.email": 1,
        "avatar": 1,
        "status": 1,
        "score": 1,
        "test_score": 1,
        "testScore": 1,
        "rank": 1,
        "category": 1,
        "predicted_role": 1,
        "job_role": 1,
        "updatedAt": 1,
        "createdAt": 1,
        "total_score": 1,
    }
    items: List[CandidateOut] = []
    cursor = db.parsed_resumes.find({"ownerUserId": owner_id}, projection=projection).sort("updatedAt", -1).limit(limit)
    async for doc in cursor:
        items.append(_normalize_candidate(doc))
    return {"items": items, "count": len(items)}
