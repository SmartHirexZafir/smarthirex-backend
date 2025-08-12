# app/routers/interviews_router.py
from __future__ import annotations

import os
import uuid
import traceback
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, EmailStr, field_validator

# --- Optional/soft imports from your existing utils ---
try:
    from app.utils.mongo import get_db as _get_db  # expected to return a pymongo.database.Database
except Exception:  # pragma: no cover
    _get_db = None  # we'll fall back to a local connector if not present

try:
    from app.utils.emailer import send_interview_invite as _send_interview_invite
except Exception:  # pragma: no cover
    _send_interview_invite = None  # we'll format HTML locally and expect your emailer util to send

try:
    from app.utils.redirect_helper import build_meeting_url as _build_meeting_url
except Exception:  # pragma: no cover
    _build_meeting_url = None

# For timezone-safe conversions using stdlib (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # if missing, we won't localize in email text

router = APIRouter(prefix="/interviews", tags=["interviews"])


# -----------------------------
# Models
# -----------------------------
class ScheduleInterviewRequest(BaseModel):
    candidate_id: str = Field(alias="candidateId")
    email: EmailStr
    starts_at: datetime = Field(alias="startsAt")  # Expecting ISO-8601 (UTC) from frontend
    timezone: str
    duration_mins: int = Field(alias="durationMins", ge=1, le=480)
    title: str
    notes: Optional[str] = None
    candidate_name: Optional[str] = None  # optional, for email copy

    # Allow camelCase keys
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "candidateId": "66b4c6d1f3a8fa0d8f9e1b2c",
                    "email": "candidate@example.com",
                    "startsAt": "2025-08-20T09:30:00Z",
                    "timezone": "Asia/Karachi",
                    "durationMins": 60,
                    "title": "Interview with SmartHirex",
                    "notes": "Bring a portfolio.",
                    "candidate_name": "Ali Raza",
                }
            ]
        },
    }

    @field_validator("starts_at", mode="before")
    @classmethod
    def _parse_starts_at(cls, v: Any) -> Any:
        # Accept strings with trailing 'Z' (replace with +00:00)
        if isinstance(v, str) and v.endswith("Z"):
            return v[:-1] + "+00:00"
        return v

    @field_validator("starts_at")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        # Normalize to UTC (tz-aware), then we'll store UTC-naive for Mongo compatibility
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("timezone")
    @classmethod
    def valid_tz(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("timezone is required")
        # Best-effort validation if ZoneInfo available
        if ZoneInfo:
            try:
                ZoneInfo(v)
            except Exception as e:
                raise ValueError(f"Invalid IANA timezone: {v}") from e
        return v


class ScheduleInterviewResponse(BaseModel):
    meeting_url: Optional[str] = Field(default=None, alias="meetingUrl")
    meeting_id: str = Field(alias="meetingId")
    status: str = "scheduled"

    model_config = {"populate_by_name": True}


class MeetingOut(BaseModel):
    id: str = Field(alias="_id")
    candidate_id: str
    email: EmailStr
    starts_at: datetime
    timezone: str
    duration_mins: int
    title: str
    notes: Optional[str] = None
    status: str
    meeting_url: Optional[str] = None
    created_at: datetime

    model_config = {"populate_by_name": True}


# -----------------------------
# Helpers
# -----------------------------
def get_db():
    """
    Dependency to obtain a Mongo database handle.
    Prefers app.utils.mongo.get_db() if available, otherwise falls back to env-based connection.
    """
    if _get_db:
        return _get_db()

    # Fallback: direct connection using pymongo (with Atlas TLS if available)
    try:
        from pymongo import MongoClient  # type: ignore
        import certifi  # type: ignore
        tls_kwargs = {"tlsCAFile": certifi.where()}
    except Exception:  # pragma: no cover
        from pymongo import MongoClient  # type: ignore
        tls_kwargs = {}

    mongo_uri = (
        os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    )
    db_name = os.getenv("MONGO_DB_NAME") or os.getenv("MONGO_DB") or "smarthirex"

    client = MongoClient(mongo_uri, **tls_kwargs)
    return client[db_name]


def build_meeting_url(token: str) -> str:
    if _build_meeting_url:
        try:
            return _build_meeting_url(token)
        except Exception:
            pass
    base = os.getenv("FRONTEND_BASE_URL") or os.getenv("WEB_BASE_URL") or "http://localhost:3000"
    return f"{base.rstrip('/')}/meetings/{token}"


def format_local_dt(dt_utc: datetime, tz_name: str) -> str:
    """
    Format datetime into a human-friendly string in the specified timezone for email copy.
    """
    try:
        if ZoneInfo:
            local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
            # Example: Tue, 20 Aug 2025 · 02:30 PM PKT
            return local_dt.strftime("%a, %d %b %Y · %I:%M %p %Z")
    except Exception:
        pass
    # Fallback to UTC string
    return dt_utc.strftime("%a, %d %b %Y · %H:%M UTC")


def default_invite_html(
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


def send_invite_email(
    email: str,
    subject: str,
    html_body: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Thin wrapper around your emailer util. If you have a richer API (templates, attachments),
    update this function only.
    """
    if _send_interview_invite:
        # If your util expects a different signature, adjust here.
        return _send_interview_invite(
            to=email,
            subject=subject,
            html=html_body,
            meta=metadata or {},
        )
    # If no util is wired, fail explicitly so the client knows.
    raise RuntimeError("Email utility not available. Implement app.utils.emailer.send_interview_invite().")


# -----------------------------
# Routes
# -----------------------------
@router.post(
    "/schedule",
    response_model=ScheduleInterviewResponse,
    status_code=status.HTTP_201_CREATED,
)
def schedule_interview(req: ScheduleInterviewRequest, db=Depends(get_db)):
    """
    Create a meeting record, generate a join URL, and send the invite email to the candidate.
    """
    now_utc = datetime.now(timezone.utc)
    if req.starts_at <= now_utc:
        raise HTTPException(status_code=400, detail="startsAt must be in the future.")

    token = uuid.uuid4().hex
    meeting_url = build_meeting_url(token)

    # Store UTC-naive datetime in Mongo (PyMongo friendly)
    starts_at_utc_naive = req.starts_at.astimezone(timezone.utc).replace(tzinfo=None)

    doc = {
        "candidate_id": req.candidate_id,
        "email": str(req.email),
        "starts_at": starts_at_utc_naive,
        "timezone": req.timezone,
        "duration_mins": req.duration_mins,
        "title": req.title,
        "notes": req.notes,
        "status": "scheduled",
        "token": token,
        "meeting_url": meeting_url,
        "created_at": now_utc.replace(tzinfo=None),
        "email_sent": False,
    }

    try:
        res = db["meetings"].insert_one(doc)
        meeting_id = str(res.inserted_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    # Prepare email
    local_dt_str = format_local_dt(req.starts_at, req.timezone)
    subject = (req.title or "Interview Invitation").strip() or "Interview Invitation"
    html_body = default_invite_html(
        candidate_name=req.candidate_name,
        title=subject,
        local_datetime_str=local_dt_str,
        meeting_url=meeting_url,
        notes=req.notes,
        timezone_name=req.timezone,
        duration_mins=req.duration_mins,
    )

    # Support dry-run to avoid SMTP errors in dev
    dry_run = os.getenv("INVITES_DRY_RUN", "0").lower() in {"1", "true", "yes"}

    # Try sending email; on failure, mark the record accordingly and report 500
    try:
        if not dry_run:
            send_invite_email(
                email=str(req.email),
                subject=subject,
                html_body=html_body,
                metadata={
                    "candidate_id": req.candidate_id,
                    "meeting_id": meeting_id,
                    "starts_at_utc": req.starts_at.isoformat(),
                    "timezone": req.timezone,
                    "duration_mins": req.duration_mins,
                    "token": token,
                },
            )
            db["meetings"].update_one(
                {"_id": res.inserted_id},
                {"$set": {"email_sent": True, "email_sent_at": datetime.utcnow()}},
            )
        else:
            db["meetings"].update_one(
                {"_id": res.inserted_id},
                {"$set": {"email_sent": False, "email_error": "DRY_RUN", "status": "scheduled"}},
            )
    except Exception as e:
        traceback.print_exc()
        db["meetings"].update_one(
            {"_id": res.inserted_id},
            {"$set": {"email_sent": False, "email_error": str(e), "status": "scheduled_email_failed"}},
        )
        raise HTTPException(status_code=500, detail=f"Failed to send invite email: {e}")

    # IMPORTANT: Use field names, not aliases, when constructing the response
    return ScheduleInterviewResponse(meeting_url=meeting_url, meeting_id=meeting_id, status="scheduled")


@router.get("/by-candidate/{candidate_id}", response_model=List[MeetingOut])
def list_interviews_for_candidate(candidate_id: str, db=Depends(get_db)):
    """
    Return meetings for a specific candidate, most recent first.
    """
    try:
        cur = db["meetings"].find({"candidate_id": candidate_id}).sort("starts_at", -1)
        items: List[Dict[str, Any]] = []
        for m in cur:
            # Normalize
            mid = str(m.get("_id"))
            items.append(
                {
                    "_id": mid,
                    "candidate_id": m.get("candidate_id"),
                    "email": m.get("email"),
                    "starts_at": m.get("starts_at"),
                    "timezone": m.get("timezone"),
                    "duration_mins": m.get("duration_mins"),
                    "title": m.get("title"),
                    "notes": m.get("notes"),
                    "status": m.get("status"),
                    "meeting_url": m.get("meeting_url"),
                    "created_at": m.get("created_at"),
                }
            )
        return items
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")
