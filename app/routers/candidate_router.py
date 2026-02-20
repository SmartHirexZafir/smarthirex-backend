# app/routers/candidate_router.py
from fastapi import APIRouter, HTTPException, Path, Body, Depends
from typing import Literal, Optional, Any, Dict, TypedDict, List
from app.utils.mongo import db
from app.utils.datetime_serialization import serialize_utc

# ✅ Auth (ownership scoping)
from app.routers.auth_router import get_current_user

# ✅ Scheduling / validation
from pydantic import BaseModel, Field, EmailStr, field_validator
from datetime import datetime, timezone
import uuid

try:
    # Optional helpers if already present in your project
    from app.utils.emailer import send_interview_invite as _send_interview_invite  # type: ignore
except Exception:
    _send_interview_invite = None  # type: ignore

try:
    from app.utils.redirect_helper import build_meeting_url as _build_meeting_url  # type: ignore
except Exception:
    _build_meeting_url = None  # type: ignore

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore

router = APIRouter()


def _job_role_of(doc: Dict[str, Any]) -> str:
    """Best-effort job role normalization used across UI."""
    return (
        doc.get("job_role")
        or doc.get("category")
        or doc.get("predicted_role")
        or (doc.get("resume") or {}).get("role")
        or "General"
    )


def _email_of(doc: Dict[str, Any]) -> str:
    return (
        (doc.get("email") or "")
        or ((doc.get("resume") or {}).get("email") or "")
    ).strip()


def _num(v: Any) -> Optional[float]:
    """Coerce possibly-string numbers to float; return None when not parseable."""
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip() != "":
            return float(v.strip())
    except Exception:
        return None
    return None


@router.get("/{candidate_id}")
async def get_candidate(
    candidate_id: str = Path(...),
    current_user: Any = Depends(get_current_user),
):
    """
    Fetch full candidate detail by ID (scoped to owner).
    Ensures all expected frontend fields are present with safe defaults.
    Also includes:
      - match_score / final_score / prompt_matching_score (coalesced to numbers)
      - test_score / testScore (normalized 0–100)
      - skills as an array
      - resume.url populated from resume_url when missing
      - attempts[] list if test submissions exist (with linkable pdf/report URLs)
    """
    owner_id = str(getattr(current_user, "id", None))
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # ---------------- Profile/defaults (no nulls) ----------------
    candidate["name"] = candidate.get("name") or "No Name"
    candidate["location"] = candidate.get("location") or "N/A"
    candidate["currentRole"] = candidate.get("currentRole") or "N/A"
    candidate["company"] = candidate.get("company") or "N/A"
    candidate["avatar"] = candidate.get("avatar") or ""

    # Resume object & defaults
    resume = candidate.setdefault("resume", {})
    resume.setdefault("summary", "")
    resume.setdefault("education", [])
    resume.setdefault("workHistory", [])
    resume.setdefault("projects", [])
    resume.setdefault("filename", candidate.get("filename", "resume.pdf"))
    # ensure a downloadable URL lives under resume.url
    resume.setdefault("url", candidate.get("resume_url", "") or candidate.get("resumeUrl", "") or "")

    # ---------------- Scores & skills normalization ----------------
    # Legacy overall match score field
    legacy_match = _num(candidate.get("score")) or 0.0

    # Normalize/coalesce the three score fields the UI may read
    match_score = _num(candidate.get("match_score"))
    final_score = _num(candidate.get("final_score") or candidate.get("finalScore"))
    prompt_matching_score = _num(candidate.get("prompt_matching_score") or candidate.get("promptMatchingScore"))

    # If any are missing, backfill from the others (then fall back to legacy score)
    if match_score is None:
        match_score = final_score if final_score is not None else (prompt_matching_score if prompt_matching_score is not None else legacy_match)
    if final_score is None:
        final_score = match_score if match_score is not None else (prompt_matching_score if prompt_matching_score is not None else legacy_match)
    if prompt_matching_score is None:
        prompt_matching_score = final_score if final_score is not None else (match_score if match_score is not None else legacy_match)

    # Clamp to 0..100
    def _clamp01(x: Optional[float]) -> float:
        if x is None:
            return 0.0
        try:
            return max(0.0, min(100.0, float(x)))
        except Exception:
            return 0.0

    match_score = _clamp01(match_score)
    final_score = _clamp01(final_score)
    prompt_matching_score = _clamp01(prompt_matching_score)
    legacy_match = _clamp01(legacy_match)

    # Persist normalized numbers back to the response shape
    candidate["match_score"] = match_score
    candidate["final_score"] = final_score
    candidate["prompt_matching_score"] = prompt_matching_score
    candidate["score"] = legacy_match  # keep legacy for existing UI bits

    # Surface both snake_case and camelCase for test score (0–100)
    test_score_val = _num(candidate.get("test_score"))
    if test_score_val is None:
        test_score_val = _num(candidate.get("testScore")) or 0.0
    candidate["test_score"] = _clamp01(test_score_val)
    candidate["testScore"] = candidate["test_score"]

    # Arrays that should never be null
    candidate["skills"] = candidate.get("skills") or []
    candidate["matchedSkills"] = candidate.get("matchedSkills") or []
    candidate["missingSkills"] = candidate.get("missingSkills") or []
    candidate["strengths"] = candidate.get("strengths") or []
    candidate["redFlags"] = candidate.get("redFlags") or []
    candidate["selectionReason"] = candidate.get("selectionReason") or ""
    candidate["rank"] = candidate.get("rank", 0)

    # Model output fields
    candidate["category"] = candidate.get("category") or candidate.get("predicted_role") or ""
    candidate["confidence"] = _num(candidate.get("confidence")) or 0.0
    candidate["match_reason"] = candidate.get("match_reason") or "ML classified"
    candidate["semantic_score"] = _num(candidate.get("semantic_score"))

    # Ensure job_role is always present for UI (Meetings/Test pages use it)
    candidate["job_role"] = _job_role_of(candidate)

    # Compute/echo total_score if missing (avg of match + test)
    if "total_score" not in candidate or candidate.get("total_score") is None:
        try:
            m = float(candidate.get("final_score") or candidate.get("match_score") or candidate.get("score") or 0)
            t = float(candidate.get("test_score") or 0)
            candidate["total_score"] = round(((m + t) / 2.0), 1)
        except Exception:
            candidate["total_score"] = None

    # ---------------- Attempts list (linkable URLs if stored) ----------------
    # If your project stores test attempts in test_submissions, expose a compact list for the History tab.
    attempts: List[Dict[str, Any]] = []
    try:
        cursor = db.test_submissions.find({"candidateId": candidate_id}).sort("createdAt", -1)
        async for a in cursor:
            raw_score = a.get("score") or a.get("result_score") or a.get("test_score")
            parsed_score = None
            n = _num(raw_score)
            if n is not None:
                parsed_score = int(round(_clamp01(n)))
            attempts.append({
                "id": str(a.get("_id") or a.get("id") or uuid.uuid4().hex),
                "submittedAt": a.get("submittedAt") or a.get("createdAt") or a.get("created_at") or a.get("timestamp"),
                "score": parsed_score,
                "pdfUrl": a.get("pdfUrl") or a.get("pdf_url") or a.get("reportUrl") or a.get("report_url") or "",
            })
    except Exception:
        # Silently ignore if the collection doesn't exist
        attempts = []

    candidate["attempts"] = attempts

    return candidate


@router.patch("/{candidate_id}/status")
async def update_candidate_status(
    candidate_id: str = Path(...),
    status: Literal["shortlisted", "rejected", "new", "interviewed", "accepted"] = Body(..., embed=True),
    current_user: Any = Depends(get_current_user),
):
    """
    Update candidate status (scoped to owner).
    Also (re)computes and persists `total_score` = avg(match score, test_score)
    so the Dashboard can list candidates under Accepted/Rejected with totals.
    """
    owner_id = str(getattr(current_user, "id", None))
    cand = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Compute total_score based on current fields
    # Prefer normalized final/match scores if available
    match_score = (
        _num(cand.get("final_score"))
        or _num(cand.get("match_score"))
        or _num(cand.get("score"))
        or 0.0
    )
    test_score = _num(cand.get("test_score")) or _num(cand.get("testScore")) or 0.0
    total_score = round(((float(match_score) + float(test_score)) / 2.0), 1)

    result = await db.parsed_resumes.update_one(
        {"_id": candidate_id, "ownerUserId": owner_id},
        {"$set": {"status": status, "total_score": total_score}},
    )

    # In Mongo, modified_count may be 0 if value was already the same.
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {"message": f"Candidate status set to '{status}'.", "total_score": total_score}


# ──────────────────────────────────────────────────────────────────────────────
# ✅ NEW: Candidates with tests (for Meetings "choose candidate" list)
#     GET /candidates/with-tests
#     Returns only candidates owned by the user that have a test_score or at least
#     one submission entry. Keeps the payload compact for dropdowns.
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/candidates/with-tests")
async def list_candidates_with_tests(current_user: Any = Depends(get_current_user)):
    owner_id = str(getattr(current_user, "id", None))

    # Collect candidateIds that have submissions
    try:
        # Not all Mongo deployments enable distinct; fallback handled below.
        submitted_ids: List[str] = await db.test_submissions.distinct("candidateId")
    except Exception:
        submitted_ids = []

    query: Dict[str, Any] = {
        "ownerUserId": owner_id,
        "$or": [
            {"test_score": {"$exists": True}},
            {"_id": {"$in": submitted_ids}} if submitted_ids else {"_id": {"$exists": True, "$in": []}},
        ],
    }

    projection = {
        "_id": 1,
        "name": 1,
        "email": 1,
        "resume.email": 1,
        "avatar": 1,
        "category": 1,
        "predicted_role": 1,
        "job_role": 1,
        "test_score": 1,
    }

    cursor = db.parsed_resumes.find(query, projection=projection).sort("updatedAt", -1)
    out: List[Dict[str, Any]] = []
    async for c in cursor:
        out.append({
            "_id": c["_id"],
            "name": c.get("name"),
            "email": _email_of(c),
            "avatar": c.get("avatar"),
            "job_role": _job_role_of(c),
            "test_score": c.get("test_score"),
        })

    return {"candidates": out}


# ──────────────────────────────────────────────────────────────────────────────
# ✅ NEW: Schedule Interview (scoped to a candidate)
#     POST /candidate/{candidate_id}/schedule
#     Motor (async Mongo) friendly, with email + meeting URL support.
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleInterviewBody(BaseModel):
    """
    Body expected from the frontend ScheduleInterviewModal.tsx.
    camelCase aliases kept for parity with UI service.
    """
    email: Optional[EmailStr] = None
    starts_at: datetime = Field(alias="startsAt")               # ISO-8601, converted to UTC
    timezone: str                                               # IANA TZ, e.g., "Asia/Karachi"
    duration_mins: int = Field(alias="durationMins", ge=1, le=480)
    title: str
    notes: Optional[str] = None
    candidate_name: Optional[str] = None

    model_config = {"populate_by_name": True}

    @field_validator("starts_at")
    @classmethod
    def _normalize_to_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("timezone")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        if not v:
            raise ValueError("timezone is required")
        if ZoneInfo:
            try:
                ZoneInfo(v)
            except Exception as e:
                raise ValueError(f"Invalid IANA timezone: {v}") from e
        return v


def _build_meeting_url_fallback(token: str) -> str:
    base = "http://localhost:3000"
    try:
        import os
        base = os.getenv("FRONTEND_BASE_URL") or os.getenv("WEB_BASE_URL") or base
    except Exception:
        pass
    return f"{base.rstrip('/')}/meetings/{token}"


async def _send_invite_email(
    to: str,
    subject: str,
    html_body: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Thin async wrapper around your email utility.
    If your util is sync, run in a thread to avoid blocking the event loop.
    """
    if _send_interview_invite is None:
        raise RuntimeError("Email utility not available. Implement app.utils.emailer.send_interview_invite().")

    try:
        import anyio
        await anyio.to_thread.run_sync(
            _send_interview_invite,
            to,
            subject,
            html_body,
            meta or {},
        )
    except ImportError:
        # anyio not present; fallback to direct call (may block)
        _send_interview_invite(to, subject, html_body, meta or {})


def _format_local_dt(dt_utc: datetime, tz_name: str) -> str:
    try:
        if ZoneInfo:
            local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
            return local_dt.strftime("%a, %d %b %Y · %I:%M %p %Z")
    except Exception:
        pass
    return dt_utc.strftime("%a, %d %b %Y · %H:%M UTC")


def _invite_html(
    candidate_name: Optional[str],
    title: str,
    local_datetime_str: str,
    meeting_url: str,
    notes: Optional[str],
    timezone_name: str,
    duration_mins: int,
) -> str:
    who = candidate_name or "Candidate"
    safe_notes = f"<p><strong>Notes:</strong> {notes}</p>" if notes else ""
    return f"""
<p>Hi <strong>{who}</strong>,</p>
<p>You’re invited to an interview.</p>
<p><strong>Title:</strong> {title}<br/>
<strong>When:</strong> {local_datetime_str} ({timezone_name})<br/>
<strong>Duration:</strong> {duration_mins} minutes</p>
{safe_notes}
<p>Join using this link:<br/>
<a href="{meeting_url}">{meeting_url}</a></p>
<p>Best,<br/>SmartHirex Team</p>
""".strip()


class ScheduleResp(TypedDict, total=False):
    meetingUrl: str
    meetingId: str
    status: str


@router.post("/{candidate_id}/schedule")
async def schedule_interview_for_candidate(
    candidate_id: str = Path(...),
    body: ScheduleInterviewBody = Body(...),
    current_user: Any = Depends(get_current_user),
) -> ScheduleResp:
    """
    Create a meeting record for this candidate, generate a meeting URL,
    and send the invite email (all scoped to the current owner).
    """
    owner_id = str(getattr(current_user, "id", None))
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    email = (body.email or candidate.get("email") or (candidate.get("resume") or {}).get("email") or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Candidate email not found. Provide 'email' in request body.")

    now_utc = datetime.now(timezone.utc)
    if body.starts_at <= now_utc:
        raise HTTPException(status_code=400, detail="startsAt must be in the future.")

    token = uuid.uuid4().hex

    # Build meeting URL (prefer project helper)
    if _build_meeting_url:
        try:
            meeting_url = _build_meeting_url(token)  # type: ignore
        except Exception:
            meeting_url = _build_meeting_url_fallback(token)
    else:
        meeting_url = _build_meeting_url_fallback(token)

    # Persist meeting (Motor async)
    doc = {
        "candidate_id": candidate_id,
        "email": email,
        "starts_at": body.starts_at,  # aware UTC datetime
        "timezone": body.timezone,
        "duration_mins": body.duration_mins,
        "title": body.title,
        "notes": body.notes,
        "status": "scheduled",
        "token": token,
        "meeting_url": meeting_url,
        "created_at": now_utc,
        "email_sent": False,
        # Auditing / scoping
        "ownerUserId": owner_id,
    }

    try:
        insert_res = await db.meetings.insert_one(doc)
        meeting_id = str(insert_res.inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    # Prepare & send email
    local_dt_str = _format_local_dt(body.starts_at, body.timezone)
    subject = (body.title or "Interview Invitation").strip()
    html_body = _invite_html(
        candidate_name=body.candidate_name or candidate.get("name"),
        title=subject,
        local_datetime_str=local_dt_str,
        meeting_url=meeting_url,
        notes=body.notes,
        timezone_name=body.timezone,
        duration_mins=body.duration_mins,
    )

    try:
        await _send_invite_email(
            to=email,
            subject=subject,
            html_body=html_body,
            meta={
                "candidate_id": candidate_id,
                "meeting_id": meeting_id,
                "starts_at_utc": serialize_utc(body.starts_at),
                "timezone": body.timezone,
                "duration_mins": body.duration_mins,
                "token": token,
                "ownerUserId": owner_id,
            },
        )
        await db.meetings.update_one(
            {"_id": insert_res.inserted_id},
            {"$set": {"email_sent": True, "email_sent_at": datetime.now(timezone.utc)}},
        )
    except Exception as e:
        await db.meetings.update_one(
            {"_id": insert_res.inserted_id},
            {"$set": {"email_sent": False, "email_error": str(e), "status": "scheduled_email_failed"}},
        )
        raise HTTPException(status_code=500, detail=f"Failed to send invite email: {e}")

    return {"meetingUrl": meeting_url, "meetingId": meeting_id, "status": "scheduled"}


# ──────────────────────────────────────────────────────────────────────────────
# ✅ NEW: Unified schedule endpoint used by UI
#     POST /interviews/schedule
#     Accepts a simpler payload and delegates to the same persistence/email flow.
#     This matches the client call pattern with { candidateId, candidateEmail, role, when, ... }.
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedScheduleBody(BaseModel):
    candidateId: str
    candidateEmail: Optional[EmailStr] = None
    candidateName: Optional[str] = None
    role: Optional[str] = None
    when: str  # ISO datetime string (UTC or with offset)
    timezone: Optional[str] = None  # optional; default UTC
    durationMins: Optional[int] = Field(default=45, ge=1, le=480)
    title: Optional[str] = None
    notes: Optional[str] = None
    dateLabel: Optional[str] = None  # passthrough (unused persistence)
    timeLabel: Optional[str] = None  # passthrough (unused persistence)

def _parse_when_to_utc_iso(when: str) -> datetime:
    try:
        # support trailing 'Z'
        iso = when.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid 'when' datetime: {e}")

@router.post("/interviews/schedule")
async def schedule_interview_unified(
    body: UnifiedScheduleBody,
    current_user: Any = Depends(get_current_user),
) -> ScheduleResp:
    owner_id = str(getattr(current_user, "id", None))
    candidate = await db.parsed_resumes.find_one({"_id": body.candidateId, "ownerUserId": owner_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    starts_at = _parse_when_to_utc_iso(body.when)
    now_utc = datetime.now(timezone.utc)
    if starts_at <= now_utc:
        raise HTTPException(status_code=400, detail="'when' must be in the future")

    email = (body.candidateEmail or _email_of(candidate))
    if not email:
        raise HTTPException(status_code=400, detail="Candidate email required")

    tz = body.timezone or "UTC"
    if ZoneInfo:
        try:
            ZoneInfo(tz)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {tz}") from e

    token = uuid.uuid4().hex
    if _build_meeting_url:
        try:
            meeting_url = _build_meeting_url(token)  # type: ignore
        except Exception:
            meeting_url = _build_meeting_url_fallback(token)
    else:
        meeting_url = _build_meeting_url_fallback(token)

    # Persist meeting
    doc = {
        "candidate_id": body.candidateId,
        "email": email,
        "starts_at": starts_at,
        "timezone": tz,
        "duration_mins": int(body.durationMins or 45),
        "title": (body.title or f"Interview with {candidate.get('name') or email}").strip(),
        "notes": body.notes,
        "status": "scheduled",
        "token": token,
        "meeting_url": meeting_url,
        "created_at": now_utc,
        "email_sent": False,
        "ownerUserId": owner_id,
        # Convenience for UI/debug
        "role": body.role or _job_role_of(candidate),
        "date_label": body.dateLabel,
        "time_label": body.timeLabel,
    }
    try:
        insert_res = await db.meetings.insert_one(doc)
        meeting_id = str(insert_res.inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    # Email
    local_str = _format_local_dt(starts_at, tz)
    subject = doc["title"]
    html_body = _invite_html(
        candidate_name=body.candidateName or candidate.get("name"),
        title=subject,
        local_datetime_str=local_str,
        meeting_url=meeting_url,
        notes=body.notes,
        timezone_name=tz,
        duration_mins=int(body.durationMins or 45),
    )

    try:
        await _send_invite_email(
            to=email,
            subject=subject,
            html_body=html_body,
            meta={
                "candidate_id": body.candidateId,
                "meeting_id": meeting_id,
                "starts_at_utc": serialize_utc(starts_at),
                "timezone": tz,
                "duration_mins": int(body.durationMins or 45),
                "token": token,
                "ownerUserId": owner_id,
            },
        )
        await db.meetings.update_one(
            {"_id": insert_res.inserted_id},
            {"$set": {"email_sent": True, "email_sent_at": datetime.now(timezone.utc)}},
        )
    except Exception as e:
        await db.meetings.update_one(
            {"_id": insert_res.inserted_id},
            {"$set": {"email_sent": False, "email_error": str(e), "status": "scheduled_email_failed"}},
        )
        raise HTTPException(status_code=500, detail=f"Failed to send invite email: {e}")

    return {"meetingUrl": meeting_url, "meetingId": meeting_id, "status": "scheduled"}
