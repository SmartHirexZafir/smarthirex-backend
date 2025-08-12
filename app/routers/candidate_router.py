# app/routers/candidate_router.py
from fastapi import APIRouter, HTTPException, Path, Body
from typing import Literal, Optional, Any, Dict
from app.utils.mongo import db

# ✅ NEW imports for scheduling
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime, timezone
import uuid
from typing import TypedDict

try:
    # Optional helpers if you already have them in your project
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


@router.get("/{candidate_id}")
async def get_candidate(candidate_id: str = Path(...)):
    """
    Fetch full candidate detail by ID.
    Ensures all expected frontend fields are populated, even with defaults.
    """
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Resume object & defaults
    candidate.setdefault("resume", {})
    resume = candidate["resume"]

    resume.setdefault("summary", "")
    resume.setdefault("education", [])
    resume.setdefault("workHistory", [])
    resume.setdefault("projects", [])
    resume.setdefault("filename", candidate.get("filename", "resume.pdf"))
    resume.setdefault("url", candidate.get("resume_url", ""))  # Optional: download link

    # Analysis fields
    candidate["score"] = candidate.get("score", 0)
    candidate["testScore"] = candidate.get("testScore", 0)
    candidate["matchedSkills"] = candidate.get("matchedSkills", [])
    candidate["missingSkills"] = candidate.get("missingSkills", [])
    candidate["strengths"] = candidate.get("strengths", [])
    candidate["redFlags"] = candidate.get("redFlags", [])
    candidate["selectionReason"] = candidate.get("selectionReason", "")
    candidate["rank"] = candidate.get("rank", 0)

    # Profile info
    candidate["name"] = candidate.get("name", "Unnamed Candidate")
    candidate["location"] = candidate.get("location", "N/A")
    candidate["currentRole"] = candidate.get("currentRole", "N/A")
    candidate["company"] = candidate.get("company", "N/A")
    candidate["avatar"] = candidate.get("avatar", "")

    # Model output fields
    candidate["category"] = candidate.get("category") or candidate.get("predicted_role", "")
    candidate["confidence"] = candidate.get("confidence", 0)
    candidate["match_reason"] = candidate.get("match_reason", "ML classified")
    candidate["semantic_score"] = candidate.get("semantic_score", None)

    return candidate


@router.patch("/{candidate_id}/status")
async def update_candidate_status(
    candidate_id: str = Path(...),
    status: Literal["shortlisted", "rejected", "new", "interviewed"] = Body(..., embed=True)
):
    """
    Update candidate status (shortlisted, rejected, etc).
    """
    result = await db.parsed_resumes.update_one(
        {"_id": candidate_id},
        {"$set": {"status": status}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Candidate not found or status unchanged")

    return {"message": f"Candidate status updated to '{status}'."}


# ──────────────────────────────────────────────────────────────────────────────
# ✅ NEW: Schedule Interview (scoped to a candidate)
#     POST /candidate/{candidate_id}/schedule
#     This mirrors the discussed flow and works with Motor (async Mongo).
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleInterviewBody(BaseModel):
    """
    Body expected from the frontend ScheduleInterviewModal.tsx
    (camelCase aliases kept for parity with UI service).
    """
    email: Optional[EmailStr] = None
    starts_at: datetime = Field(alias="startsAt")               # ISO-8601, treated/converted to UTC
    timezone: str                                               # IANA TZ, e.g., "Asia/Karachi"
    duration_mins: int = Field(alias="durationMins", ge=1, le=480)
    title: str
    notes: Optional[str] = None
    candidate_name: Optional[str] = None

    model_config = {"populate_by_name": True}

    @validator("starts_at")
    def normalize_to_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @validator("timezone")
    def validate_tz(cls, v: str) -> str:
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
    Thin async wrapper around your email utility. If your util is sync, we'll
    run it in a thread so we don't block the event loop.
    """
    if _send_interview_invite is None:
        raise RuntimeError("Email utility not available. Implement app.utils.emailer.send_interview_invite().")

    # Run sync function without blocking the event loop
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
) -> ScheduleResp:
    """
    Create a meeting record for this candidate, generate a meeting URL,
    and send the invite email.
    """
    # Ensure the candidate exists and fetch fallback email if not provided
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    email = (body.email or candidate.get("email") or (candidate.get("resume") or {}).get("email") or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Candidate email not found. Provide 'email' in request body.")

    now_utc = datetime.now(timezone.utc)
    if body.starts_at <= now_utc:
        raise HTTPException(status_code=400, detail="startsAt must be in the future.")

    token = uuid.uuid4().hex
    # Use project helper if available
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
                "starts_at_utc": body.starts_at.isoformat(),
                "timezone": body.timezone,
                "duration_mins": body.duration_mins,
                "token": token,
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
