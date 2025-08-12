# app/services/interviews_service.py
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

# Schemas (unchanged)
from app.schemas.interviews import (
    ScheduleInterviewRequest,
    ScheduleInterviewResponse,
    MeetingOut,
)

# Optional utilities (soft import so the service stays testable)
try:
    from app.utils.emailer import send_interview_invite as _send_interview_invite
except Exception:  # pragma: no cover
    _send_interview_invite = None  # type: ignore

try:
    from app.utils.redirect_helper import build_meeting_url as _build_meeting_url
except Exception:  # pragma: no cover
    _build_meeting_url = None  # type: ignore

# Prefer stdlib zoneinfo for IANA timezone formatting
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# -----------------------------
# Helpers (pure functions)
# -----------------------------
def _format_local_dt(dt_utc: datetime, tz_name: str) -> str:
    """
    Format a UTC datetime to human-friendly string in the specified timezone.
    """
    try:
        if ZoneInfo:
            local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
            # Example: Tue, 20 Aug 2025 · 02:30 PM PKT
            return local_dt.strftime("%a, %d %b %Y · %I:%M %p %Z")
    except Exception:
        pass
    # Fallback to UTC
    return dt_utc.strftime("%a, %d %b %Y · %H:%M UTC")


def _default_invite_html(
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
{saf e_notes}
<p>Join using this link:<br/>
<a href="{meeting_url}">{meeting_url}</a></p>
<p>Best,<br/>SmartHirex Team</p>
""".strip()


def _fallback_build_meeting_url(token: str) -> str:
    """
    Frontend "gate" URL – user lands here first.
    """
    base = os.getenv("FRONTEND_BASE_URL") or os.getenv("WEB_BASE_URL") or "http://localhost:3000"
    return f"{base.rstrip('/')}/meetings/{token}"


def _build_google_meet_url(token: str) -> str:
    """
    External Google Meet (or any provider) URL. Default template is lookup-based Meet link.
    Override with env GOOGLE_MEET_URL_TEMPLATE, e.g.:
      GOOGLE_MEET_URL_TEMPLATE=https://meet.google.com/lookup/{token}
    """
    template = os.getenv("GOOGLE_MEET_URL_TEMPLATE", "https://meet.google.com/lookup/{token}")
    return template.replace("{token}", token)


# -----------------------------
# Service
# -----------------------------
class InterviewsService:
    """
    Business logic for interview scheduling.

    Parameters
    ----------
    db : Database-like
        Must expose collection access via db["meetings"] with insert_one, update_one, find, etc.
    emailer : Optional[Callable]
        Custom email function override. Signature:
            emailer(to: str, subject: str, html: str, meta: Dict[str, Any]) -> Any
        If not provided, tries to use app.utils.emailer.send_interview_invite.
    url_builder : Optional[Callable[[str], str]]
        Custom meeting URL (frontend gate) builder override. Given a token, returns full URL.
    """

    def __init__(
        self,
        db: Any,
        emailer: Optional[Callable[[str, str, str, Dict[str, Any]], Any]] = None,
        url_builder: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.db = db
        self.emailer = emailer or _send_interview_invite
        self.url_builder = url_builder or _build_meeting_url or _fallback_build_meeting_url

        if self.emailer is None:
            # Fail fast so misconfiguration is obvious in dev
            raise RuntimeError(
                "Email utility is not available. "
                "Provide 'emailer' or implement app.utils.emailer.send_interview_invite()."
            )

    # -------- Core actions --------
    def schedule(self, req: ScheduleInterviewRequest) -> ScheduleInterviewResponse:
        """
        Create a meeting record, generate a join URL (frontend gate),
        and send the invite email.

        Returns
        -------
        ScheduleInterviewResponse
        """
        now_utc = datetime.now(timezone.utc)
        if req.starts_at <= now_utc:
            # Ensure future datetime; request validator already normalized to UTC
            raise ValueError("startsAt must be in the future.")

        # Token + URLs
        token = uuid.uuid4().hex
        meeting_url = self.url_builder(token)            # your gate page (/meetings/{token})
        google_meet_url = _build_google_meet_url(token)  # external provider URL

        # Store UTC-naive fields to match router & PyMongo-friendly style
        starts_at_utc_naive = req.starts_at.astimezone(timezone.utc).replace(tzinfo=None)
        created_at_utc_naive = now_utc.replace(tzinfo=None)

        # Persist meeting
        doc = {
            "candidate_id": req.candidate_id,
            "email": str(req.email),
            "starts_at": starts_at_utc_naive,   # UTC-naive datetime
            "timezone": req.timezone,
            "duration_mins": req.duration_mins,
            "title": req.title,
            "notes": req.notes,
            "status": "scheduled",
            "token": token,
            "meeting_url": meeting_url,           # frontend gate
            "google_meet_url": google_meet_url,   # external meet link
            "external_url": google_meet_url,      # generic alias
            "created_at": created_at_utc_naive,   # UTC-naive
            "email_sent": False,
        }

        try:
            res = self.db["meetings"].insert_one(doc)
            meeting_id = str(res.inserted_id)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"DB insert failed: {e}") from e

        # Prepare email
        local_dt_str = _format_local_dt(req.starts_at, req.timezone)
        subject = (req.title or "Interview Invitation").strip()
        html_body = _default_invite_html(
            candidate_name=req.candidate_name,
            title=subject,
            local_datetime_str=local_dt_str,
            meeting_url=meeting_url,  # always send the gate URL in email
            notes=req.notes,
            timezone_name=req.timezone,
            duration_mins=req.duration_mins,
        )

        # Send email + update record
        try:
            self.emailer(
                to=str(req.email),
                subject=subject,
                html=html_body,
                meta={
                    "candidate_id": req.candidate_id,
                    "meeting_id": meeting_id,
                    "starts_at_utc": req.starts_at.isoformat(),
                    "timezone": req.timezone,
                    "duration_mins": req.duration_mins,
                    "token": token,
                    "google_meet_url": google_meet_url,
                },
            )
            self.db["meetings"].update_one(
                {"_id": res.inserted_id},
                {"$set": {"email_sent": True, "email_sent_at": datetime.now(timezone.utc)}},
            )
        except Exception as e:  # pragma: no cover
            self.db["meetings"].update_one(
                {"_id": res.inserted_id},
                {"$set": {"email_sent": False, "email_error": str(e), "status": "scheduled_email_failed"}},
            )
            raise RuntimeError(f"Failed to send invite email: {e}") from e

        return ScheduleInterviewResponse(meetingUrl=meeting_url, meetingId=meeting_id, status="scheduled")

    def list_by_candidate(self, candidate_id: str) -> List[MeetingOut]:
        """
        Fetch meetings for a candidate, most recent first.
        """
        try:
            cur = self.db["meetings"].find({"candidate_id": candidate_id}).sort("starts_at", -1)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"DB query failed: {e}") from e

        items: List[MeetingOut] = []
        for m in cur:
            m["_id"] = str(m.get("_id"))
            items.append(
                MeetingOut(
                    _id=m["_id"],
                    candidate_id=m.get("candidate_id"),
                    email=m.get("email"),
                    starts_at=m.get("starts_at"),
                    timezone=m.get("timezone"),
                    duration_mins=m.get("duration_mins"),
                    title=m.get("title"),
                    notes=m.get("notes"),
                    status=m.get("status"),
                    meeting_url=m.get("meeting_url"),
                    created_at=m.get("created_at"),
                )
            )
        return items

    # -------- Extra helper for the /meetings/[token] gate page --------
    def get_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single meeting document by token.
        Returns a dict with all fields (including google_meet_url / external_url),
        or None if not found. This is intentionally not tied to MeetingOut, so we
        don't break existing types that don't include the new fields.
        """
        if not token:
            return None
        doc = self.db["meetings"].find_one({"token": token})
        if not doc:
            return None

        # Normalize for JSON responses in routers
        _id = str(doc.get("_id"))
        starts_at = doc.get("starts_at")
        # Prepare camel + snake friendly shape if router wants it
        if isinstance(starts_at, datetime):
            if starts_at.tzinfo is None:
                starts_iso = starts_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            else:
                starts_iso = starts_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            starts_iso = None

        base = {
            "_id": _id,
            "token": doc.get("token"),
            "title": doc.get("title"),
            "email": doc.get("email"),
            "candidate_id": doc.get("candidate_id"),
            "status": doc.get("status"),
            "starts_at": starts_at,  # raw (UTC-naive) for internal uses
            "startsAt": starts_iso,  # ISO 'Z' for clients
            "timezone": doc.get("timezone"),
            "duration_mins": doc.get("duration_mins"),
            "durationMins": doc.get("duration_mins"),
            "meeting_url": doc.get("meeting_url"),
            "meetingUrl": doc.get("meeting_url"),
            "google_meet_url": doc.get("google_meet_url"),
            "googleMeetUrl": doc.get("google_meet_url"),
            "external_url": doc.get("external_url"),
            "externalUrl": doc.get("external_url"),
            "notes": doc.get("notes"),
            "created_at": doc.get("created_at"),
            "now": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return base
