# app/routers/interviews_router.py
from __future__ import annotations

import os
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field, EmailStr, field_validator
from app.routers.auth_router import get_current_user  # ✅ require auth for Meetings endpoints

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


class EligibleCandidate(BaseModel):
    _id: str
    name: Optional[str] = None
    email: Optional[str] = None
    job_role: Optional[str] = None
    test_score: Optional[float] = None
    avatar: Optional[str] = None


class TodayStats(BaseModel):
    scheduled_today: int
    timezone: str


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


def build_external_meet_url(token: str) -> str:
    """
    External Google Meet (or provider) URL. Override with:
      GOOGLE_MEET_URL_TEMPLATE="https://meet.google.com/lookup/{token}"
    """
    template = os.getenv("GOOGLE_MEET_URL_TEMPLATE", "https://meet.google.com/lookup/{token}")
    return template.replace("{token}", token)


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


def _job_role_of(doc: Dict[str, Any]) -> str:
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


def _candidate_has_test(db, candidate_id: str) -> bool:
    """
    Returns True if the candidate has a test_score OR at least one submission in test_submissions.
    """
    cand = db["parsed_resumes"].find_one({"_id": candidate_id})
    if cand and cand.get("test_score") is not None:
        return True
    # fallback: check submissions
    sub = db["test_submissions"].find_one({"candidateId": candidate_id})
    return bool(sub)


# -----------------------------
# Routes
# -----------------------------
@router.post(
    "/schedule",
    response_model=ScheduleInterviewResponse,
    status_code=status.HTTP_201_CREATED,
)
def schedule_interview(
    req: ScheduleInterviewRequest,
    db=Depends(get_db),
    current=Depends(get_current_user),  # ✅ protect scheduling
):
    """
    Create a meeting record, generate a join URL, and send the invite email to the candidate.
    The join URL is the FRONTEND "gate" page (/meetings/{token}). We also store
    an external Google Meet URL which the gate will open at the right time.

    Enhancement: by default, we enforce that a candidate must have taken a test before
    scheduling. Set ALLOW_SCHEDULE_WITHOUT_TEST=1 to bypass this check.
    """
    now_utc = datetime.now(timezone.utc)
    if req.starts_at <= now_utc:
        raise HTTPException(status_code=400, detail="startsAt must be in the future.")

    # Enforce "test before schedule" unless explicitly disabled
    allow_without_test = os.getenv("ALLOW_SCHEDULE_WITHOUT_TEST", "0").lower() in {"1", "true", "yes", "on"}
    if not allow_without_test and not _candidate_has_test(db, req.candidate_id):
        raise HTTPException(
            status_code=400,
            detail="This candidate hasn't completed a test yet. Please ask them to complete the assessment before scheduling the interview.",
        )

    token = uuid.uuid4().hex
    meeting_url = build_meeting_url(token)          # gate page
    external_url = build_external_meet_url(token)   # provider (Google Meet) URL

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
        "meeting_url": meeting_url,          # frontend gate
        "google_meet_url": external_url,     # explicit
        "external_url": external_url,        # generic alias
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
        meeting_url=meeting_url,  # always email the gate page
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
                    "google_meet_url": external_url,
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
def list_interviews_for_candidate(
    candidate_id: str,
    db=Depends(get_db),
    current=Depends(get_current_user),  # ✅ protect listing
):
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


# -------- New: gate page data for /meetings/[token] --------
@router.get("/by-token/{token}")
def get_meeting_by_token(token: str, db=Depends(get_db)):
    """
    Returns meeting details for the gate page (no auth).
    Includes BOTH snake_case and camelCase keys for compatibility with different frontends.
    """
    doc = db["meetings"].find_one({"token": token})
    if not doc:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Normalize starts_at (stored as UTC naive) to ISO 'Z'
    starts_at = doc.get("starts_at")
    if isinstance(starts_at, datetime):
        if starts_at.tzinfo is None:
            starts_iso = starts_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            starts_iso = starts_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        starts_iso = None

    # Prefer explicit google_meet_url, fallback to external_url
    external = doc.get("google_meet_url") or doc.get("external_url")

    # --- snake_case (current Next.js page.tsx expects these) ---
    data_snake = {
        "token": doc.get("token"),
        "title": doc.get("title"),
        "status": doc.get("status"),
        "email": doc.get("email"),
        "candidate_id": doc.get("candidate_id"),
        "starts_at": starts_iso,
        "timezone": doc.get("timezone"),
        "duration_mins": doc.get("duration_mins"),
        "meeting_url": doc.get("meeting_url"),
        "external_url": external,
        "google_meet_url": doc.get("google_meet_url"),
        "notes": doc.get("notes"),
        "now": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    # --- camelCase (kept for backward compatibility) ---
    data_camel = {
        "token": data_snake["token"],
        "title": data_snake["title"],
        "status": data_snake["status"],
        "email": data_snake["email"],
        "candidateId": data_snake["candidate_id"],
        "startsAt": data_snake["starts_at"],
        "timezone": data_snake["timezone"],
        "durationMins": data_snake["duration_mins"],
        "meetingUrl": data_snake["meeting_url"],
        "externalUrl": data_snake["external_url"],
        "googleMeetUrl": data_snake["google_meet_url"],
        "notes": data_snake["notes"],
        "now": data_snake["now"],
    }

    # Merge; snake_case fields take precedence for duplicate keys with same meaning
    return {**data_camel, **data_snake}


# -------- New: Eligible candidates (have taken a test) --------
@router.get("/eligible", response_model=List[EligibleCandidate])
def list_eligible_candidates(
    limit: int = Query(100, ge=1, le=500),
    db=Depends(get_db),
    current=Depends(get_current_user),  # ✅ protect eligible list
):
    """
    Returns candidates who are eligible for scheduling (i.e., have taken a test).
    Eligibility rule:
      - candidate.test_score exists, OR
      - there is at least one submission in test_submissions
    """
    # First, gather IDs with submissions
    try:
        submitted_ids = list(db["test_submissions"].distinct("candidateId"))
    except Exception:
        submitted_ids = []

    query: Dict[str, Any] = {
        "$or": [
            {"test_score": {"$exists": True}},
            {"_id": {"$in": submitted_ids}} if submitted_ids else {"_id": None},  # harmless false branch
        ]
    }
    projection = {
        "_id": 1,
        "name": 1,
        "email": 1,
        "resume.email": 1,
        "avatar": 1,
        "job_role": 1,
        "category": 1,
        "predicted_role": 1,
        "test_score": 1,
    }

    cur = db["parsed_resumes"].find(query, projection=projection).sort("updatedAt", -1).limit(limit)
    out: List[EligibleCandidate] = []
    for c in cur:
        out.append(
            EligibleCandidate(
                _id=c["_id"],
                name=c.get("name"),
                email=_email_of(c),
                job_role=_job_role_of(c),
                test_score=c.get("test_score"),
                avatar=c.get("avatar"),
            )
        )
    return out


# -------- New: Today stats (scheduled count) --------
@router.get("/stats/today", response_model=TodayStats)
def stats_today(
    tz: str = Query("UTC", description="IANA timezone (e.g., Asia/Karachi)"),
    db=Depends(get_db),
    current=Depends(get_current_user),  # ✅ protect stats endpoint
):
    """
    Returns number of meetings scheduled for 'today' in the given timezone.
    Useful for the Meetings header badges (e.g., '5 Scheduled Today').
    """
    if ZoneInfo:
        try:
            z = ZoneInfo(tz)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {tz}") from e
    else:
        # Fallback: treat tz as UTC
        z = timezone.utc  # type: ignore

    now = datetime.now(timezone.utc).astimezone(z)
    start_local = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)

    # Convert bounds to UTC-naive (to match stored 'starts_at')
    start_utc_naive = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc_naive = end_local.astimezone(timezone.utc).replace(tzinfo=None)

    count = db["meetings"].count_documents({
        "starts_at": {"$gte": start_utc_naive, "$lt": end_utc_naive},
        "status": {"$in": ["scheduled", "rescheduled"]},
    })

    return TodayStats(scheduled_today=int(count), timezone=tz)
