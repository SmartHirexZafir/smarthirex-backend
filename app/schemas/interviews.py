# app/schemas/interviews.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field, EmailStr, validator

# Prefer stdlib zoneinfo for IANA timezone validation if available
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ===============================
# Request / Response Schemas
# ===============================

class ScheduleInterviewRequest(BaseModel):
    """
    Payload sent by the frontend to schedule an interview and email the invite.

    Frontend sends camelCase keys; server models expose snake_case as well.
    """
    candidate_id: str = Field(alias="candidateId")
    email: EmailStr
    starts_at: datetime = Field(alias="startsAt")  # Expect ISO-8601 (UTC) from client
    timezone: str                                   # IANA TZ, e.g. "Asia/Karachi"
    duration_mins: int = Field(alias="durationMins", ge=1, le=480)
    title: str
    notes: Optional[str] = None
    candidate_name: Optional[str] = None

    # Allow both aliases and field names when parsing/serializing
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
                    "notes": "Bring your portfolio.",
                    "candidate_name": "Ali Raza",
                }
            ]
        },
    }

    @validator("starts_at")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Accept naive as UTC; otherwise convert to UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @validator("timezone")
    def validate_timezone(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("timezone is required")
        if ZoneInfo:
            try:
                ZoneInfo(v)
            except Exception as e:
                raise ValueError(f"Invalid IANA timezone: {v}") from e
        return v


class ScheduleInterviewResponse(BaseModel):
    """
    Response returned after successfully creating a meeting and sending invite.
    """
    meeting_url: Optional[str] = Field(default=None, alias="meetingUrl")
    meeting_id: str = Field(alias="meetingId")
    status: str = "scheduled"

    model_config = {"populate_by_name": True}


# ===============================
# DB / Output Schemas
# ===============================

class MeetingOut(BaseModel):
    """
    Public-facing meeting object (as returned from list/get endpoints).
    """
    id: str = Field(alias="_id")
    candidate_id: str
    email: EmailStr
    starts_at: str
    timezone: str
    duration_mins: int
    title: str
    notes: Optional[str] = None
    status: str
    meeting_url: Optional[str] = None
    created_at: str

    model_config = {"populate_by_name": True}


class MeetingDB(BaseModel):
    """
    Internal DB schema for the `meetings` collection. Useful if you structure
    your data access layer with typed models. Not required for FastAPI routes.
    """
    candidate_id: str
    email: EmailStr
    starts_at: datetime
    timezone: str
    duration_mins: int
    title: str
    notes: Optional[str] = None
    status: str = "scheduled"
    token: str
    meeting_url: Optional[str] = None
    created_at: datetime
    email_sent: bool = False
    email_sent_at: Optional[datetime] = None

    @classmethod
    def from_request(cls, token: str, req: ScheduleInterviewRequest) -> "MeetingDB":
        return cls(
            candidate_id=req.candidate_id,
            email=req.email,
            starts_at=req.starts_at,  # already UTC by validator
            timezone=req.timezone,
            duration_mins=req.duration_mins,
            title=req.title,
            notes=req.notes,
            token=token,
            created_at=datetime.now(timezone.utc),
        )


__all__ = [
    "ScheduleInterviewRequest",
    "ScheduleInterviewResponse",
    "MeetingOut",
    "MeetingDB",
]
