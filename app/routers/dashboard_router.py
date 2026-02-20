# app/routers/dashboard_router.py
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.routers.auth_router import get_current_user
from app.utils.datetime_serialization import serialize_utc_any
from app.utils.mongo import db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


StatusBucket = TypedDict(
    "StatusBucket",
    {"count": int, "avg_total_score": Optional[float], "avg_test_score": Optional[float]},
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
        "score": Optional[float],
        "test_score": Optional[float],
        "total_score": Optional[float],
        "rank": Optional[int],
        "updatedAt": Optional[str],
        "createdAt": Optional[str],
    },
    total=False,
)


def _owner_id_or_401(current_user: Any) -> str:
    owner_id = str(getattr(current_user, "id", None) or "")
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return owner_id


def _email_of(doc: Dict[str, Any]) -> Optional[str]:
    return doc.get("email") or (doc.get("resume") or {}).get("email") or None


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
        "updatedAt": serialize_utc_any(doc.get("updatedAt")) or doc.get("updatedAt"),
        "createdAt": serialize_utc_any(doc.get("createdAt")) or doc.get("createdAt"),
    }


def _utc_day_bounds(now_utc: datetime) -> tuple[datetime, datetime]:
    start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


async def _fetch_candidates_by_status(
    owner_id: str,
    statuses: List[str],
    limit: int,
    include_no_status: bool = False,
) -> List[CandidateOut]:
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
    cursor = db.parsed_resumes.find(query, projection=projection).limit(limit)
    async for doc in cursor:
        items.append(_normalize_candidate(doc))
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


@router.get("/summary")
async def dashboard_summary(current_user: Any = Depends(get_current_user)):
    """
    Authoritative, recruiter-scoped dashboard aggregate endpoint.
    Uses aggregation/count queries only (no frontend-side metric computation).
    """
    owner_id = _owner_id_or_401(current_user)
    now_utc = datetime.now(timezone.utc)
    today_start_utc, tomorrow_start_utc = _utc_day_bounds(now_utc)

    # Candidate funnel metrics in one aggregation (single collection roundtrip).
    candidate_pipeline = [
        {"$match": {"ownerUserId": owner_id}},
        {
            "$facet": {
                "total_candidates_uploaded": [{"$count": "count"}],
                "total_filtered_candidates": [
                    {
                        "$match": {
                            "$or": [
                                {"status": {"$in": ["new", "interviewed", "shortlisted", "rejected", "accepted", "approved"]}},
                                {"final_score": {"$exists": True}},
                                {"prompt_matching_score": {"$exists": True}},
                            ]
                        }
                    },
                    {"$count": "count"},
                ],
                "status_buckets": [
                    {
                        "$group": {
                            "_id": {"$toLower": {"$ifNull": ["$status", ""]}},
                            "count": {"$sum": 1},
                        }
                    }
                ],
            }
        },
    ]
    candidate_agg = await db.parsed_resumes.aggregate(candidate_pipeline).to_list(length=1)
    candidate_root = candidate_agg[0] if candidate_agg else {}

    status_counts: Dict[str, int] = {}
    for row in candidate_root.get("status_buckets", []):
        k = str(row.get("_id") or "").strip().lower()
        status_counts[k] = int(row.get("count") or 0)

    total_candidates_uploaded = int(
        ((candidate_root.get("total_candidates_uploaded") or [{}])[0]).get("count") or 0
    )
    total_filtered_candidates = int(
        ((candidate_root.get("total_filtered_candidates") or [{}])[0]).get("count") or 0
    )
    total_shortlisted = int(status_counts.get("shortlisted", 0))
    total_rejected = int(status_counts.get("rejected", 0))
    # Backward-compatible alias: "accepted" documents still count as approved.
    total_approved = int(status_counts.get("approved", 0) + status_counts.get("accepted", 0))
    total_awaiting_review = int(
        status_counts.get("new", 0)
        + status_counts.get("interviewed", 0)
        + status_counts.get("awaiting_review", 0)
        + status_counts.get("", 0)
    )

    owned_candidate_ids: List[str] = await db.parsed_resumes.distinct("_id", {"ownerUserId": owner_id})
    tests_match_base: Dict[str, Any] = {"candidateId": {"$in": owned_candidate_ids or []}}

    total_tests_conducted = 0
    total_tests_pending = 0
    today_tests_count = 0
    upcoming_tests: List[Dict[str, Any]] = []

    if owned_candidate_ids:
        total_tests_conducted = int(await db.test_submissions.count_documents(tests_match_base))
        total_tests_pending = int(
            await db.test_invites.count_documents(
                {**tests_match_base, "status": {"$in": ["pending", "active"]}}
            )
        )

        test_schedule_base = [
            {"$match": {**tests_match_base, "status": {"$in": ["pending", "active"]}}},
            {
                "$addFields": {
                    "_scheduled_dt_utc": {
                        "$switch": {
                            "branches": [
                                {"case": {"$eq": [{"$type": "$scheduledDateTime"}, "date"]}, "then": "$scheduledDateTime"},
                                {
                                    "case": {"$eq": [{"$type": "$scheduledDateTime"}, "string"]},
                                    "then": {"$dateFromString": {"dateString": "$scheduledDateTime", "onError": None, "onNull": None}},
                                },
                            ],
                            "default": None,
                        }
                    }
                }
            },
            {"$match": {"_scheduled_dt_utc": {"$ne": None}}},
        ]

        today_tests_pipeline = test_schedule_base + [
            {"$match": {"_scheduled_dt_utc": {"$gte": today_start_utc, "$lt": tomorrow_start_utc}}},
            {"$count": "count"},
        ]
        today_tests_rows = await db.test_invites.aggregate(today_tests_pipeline).to_list(length=1)
        today_tests_count = int((today_tests_rows[0] if today_tests_rows else {}).get("count") or 0)

        upcoming_tests_pipeline = test_schedule_base + [
            {"$match": {"_scheduled_dt_utc": {"$gte": now_utc}}},
            {"$sort": {"_scheduled_dt_utc": 1}},
            {"$limit": 5},
            {
                "$lookup": {
                    "from": "parsed_resumes",
                    "let": {"cid": "$candidateId"},
                    "pipeline": [
                        {"$match": {"$expr": {"$eq": ["$_id", "$$cid"]}}},
                        {"$project": {"_id": 1, "name": 1, "email": 1, "resume.email": 1, "job_role": 1, "category": 1, "predicted_role": 1}},
                    ],
                    "as": "_candidate",
                }
            },
            {"$addFields": {"_cand": {"$arrayElemAt": ["$_candidate", 0]}}},
            {
                "$project": {
                    "_id": 1,
                    "candidateId": 1,
                    "status": 1,
                    "scheduledDateTime": "$_scheduled_dt_utc",
                    "testDurationMinutes": 1,
                    "candidateName": "$_cand.name",
                    "candidateEmail": {"$ifNull": ["$_cand.email", "$_cand.resume.email"]},
                    "job_role": {"$ifNull": ["$_cand.job_role", {"$ifNull": ["$_cand.category", "$_cand.predicted_role"]}]},
                }
            },
        ]
        upcoming_tests = await db.test_invites.aggregate(upcoming_tests_pipeline).to_list(length=5)

    meetings_base_match = {
        "ownerUserId": owner_id,
        "status": {"$in": ["scheduled", "rescheduled"]},
    }
    today_meetings_count = int(
        await db.meetings.count_documents(
            {
                **meetings_base_match,
                "starts_at": {"$gte": today_start_utc, "$lt": tomorrow_start_utc},
            }
        )
    )
    upcoming_meetings = await db.meetings.aggregate(
        [
            {"$match": {**meetings_base_match, "starts_at": {"$gte": now_utc}}},
            {"$sort": {"starts_at": 1}},
            {"$limit": 5},
            {
                "$project": {
                    "_id": 1,
                    "candidateId": "$candidate_id",
                    "status": 1,
                    "starts_at": 1,
                    "ends_at": 1,
                    "title": 1,
                    "duration_mins": 1,
                    "timezone": 1,
                    "email": 1,
                    "meeting_url": 1,
                }
            },
        ]
    ).to_list(length=5)

    # Last 7 days trends (UTC); fill missing days with 0
    seven_days_ago_start = (now_utc - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    last_7_dates_asc = [(now_utc - timedelta(days=(6 - i))).strftime("%Y-%m-%d") for i in range(7)]

    last_7_tests: List[Dict[str, Any]] = []
    last_7_uploads: List[Dict[str, Any]] = []
    if owned_candidate_ids:
        tests_by_day = await db.test_submissions.aggregate(
            [
                {"$match": {"candidateId": {"$in": owned_candidate_ids}, "submittedAt": {"$gte": seven_days_ago_start}}},
                {"$group": {"_id": {"$dateToString": {"date": "$submittedAt", "format": "%Y-%m-%d"}}, "count": {"$sum": 1}}},
            ]
        ).to_list(length=10)
        day_to_tests = {r["_id"]: int(r.get("count") or 0) for r in tests_by_day}
        last_7_tests = [{"date": d, "count": day_to_tests.get(d, 0)} for d in last_7_dates_asc]

    uploads_by_day = await db.parsed_resumes.aggregate(
        [
            {"$match": {"ownerUserId": owner_id, "createdAt": {"$exists": True, "$gte": seven_days_ago_start}}},
            {"$group": {"_id": {"$dateToString": {"date": "$createdAt", "format": "%Y-%m-%d"}}, "count": {"$sum": 1}}},
        ]
    ).to_list(length=10)
    day_to_uploads = {r["_id"]: int(r.get("count") or 0) for r in uploads_by_day}
    last_7_uploads = [{"date": d, "count": day_to_uploads.get(d, 0)} for d in last_7_dates_asc]

    # Recent 10 activities: merge test_submitted, meeting_scheduled, status_changed; sort by at desc; limit 10
    activities: List[Dict[str, Any]] = []
    if owned_candidate_ids:
        subs = await db.test_submissions.find(
            {"candidateId": {"$in": owned_candidate_ids}},
            {"_id": 1, "candidateId": 1, "submittedAt": 1},
        ).sort("submittedAt", -1).limit(10).to_list(length=10)
        for s in subs:
            at = s.get("submittedAt")
            at_iso = serialize_utc_any(at) if at else None
            if at_iso:
                activities.append({"type": "test_submitted", "at": at_iso, "candidateId": s.get("candidateId"), "label": "Test submitted"})
    meet_docs = await db.meetings.find(
        {"ownerUserId": owner_id},
        {"_id": 1, "candidate_id": 1, "created_at": 1, "title": 1},
    ).sort("created_at", -1).limit(10).to_list(length=10)
    for m in meet_docs:
        at = m.get("created_at")
        at_iso = serialize_utc_any(at) if at else None
        if at_iso:
            activities.append({"type": "meeting_scheduled", "at": at_iso, "candidateId": m.get("candidate_id"), "label": m.get("title") or "Meeting scheduled"})
    status_docs = await db.parsed_resumes.find(
        {"ownerUserId": owner_id, "status": {"$in": ["shortlisted", "rejected", "accepted", "approved"]}},
        {"_id": 1, "name": 1, "status": 1, "updatedAt": 1},
    ).sort("updatedAt", -1).limit(10).to_list(length=10)
    for c in status_docs:
        at = c.get("updatedAt")
        at_iso = serialize_utc_any(at) if at else None
        if at_iso:
            st = (c.get("status") or "").lower()
            activities.append({"type": "status_changed", "at": at_iso, "candidateId": c.get("_id"), "label": f"Status set to {st}", "status": st})
    activities.sort(key=lambda x: x.get("at") or "", reverse=True)
    recent_activities = activities[:10]

    return JSONResponse(
        {
            "totalCandidatesUploaded": total_candidates_uploaded,
            "totalFilteredCandidates": total_filtered_candidates,
            "totalTestsConducted": total_tests_conducted,
            "totalTestsPending": total_tests_pending,
            "totalApproved": total_approved,
            "totalShortlisted": total_shortlisted,
            "totalRejected": total_rejected,
            "totalAwaitingReview": total_awaiting_review,
            "todayTestsCount": today_tests_count,
            "todayMeetingsCount": today_meetings_count,
            "upcomingTests": upcoming_tests,
            "upcomingMeetings": upcoming_meetings,
            "last7DaysTestSubmissions": last_7_tests,
            "last7DaysCvUploads": last_7_uploads,
            "recentActivities": recent_activities,
            "generatedAt": now_utc.isoformat(),
        }
    )


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

    filename = f"dashboard_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
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
