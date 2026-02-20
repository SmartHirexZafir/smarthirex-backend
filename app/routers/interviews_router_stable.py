from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import AliasChoices, BaseModel, EmailStr, Field, field_validator, model_validator

from app.routers.auth_router import get_current_user
from app.utils.datetime_serialization import serialize_utc, serialize_utc_any
from app.utils.mongo import db
from app.utils.redirect_helper import build_meeting_url

try:
    from app.utils.emailer import send_interview_invite as _send_interview_invite
except Exception:  # pragma: no cover
    _send_interview_invite = None

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

router = APIRouter(prefix="/interviews", tags=["interviews"])


class ScheduleInterviewRequest(BaseModel):
    candidate_id: str = Field(validation_alias=AliasChoices("candidateId", "candidate_id"))
    email: EmailStr = Field(validation_alias=AliasChoices("email", "candidateEmail"))
    starts_at: datetime = Field(validation_alias=AliasChoices("startsAt", "startAt", "when"))
    timezone: str
    duration_mins: int = Field(default=45, validation_alias=AliasChoices("durationMins", "duration"), ge=1, le=480)
    title: Optional[str] = None
    role: Optional[str] = None
    notes: Optional[str] = None
    candidate_name: Optional[str] = Field(default=None, validation_alias=AliasChoices("candidate_name", "candidateName"))

    model_config = {"populate_by_name": True}

    @field_validator("starts_at", mode="before")
    @classmethod
    def _parse_starts_at(cls, v: Any) -> Any:
        if isinstance(v, str) and v.endswith("Z"):
            return v[:-1] + "+00:00"
        return v

    @field_validator("starts_at")
    @classmethod
    def _must_be_aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("timezone")
    @classmethod
    def _validate_tz(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("timezone is required")
        if ZoneInfo:
            try:
                ZoneInfo(v.strip())
            except Exception as exc:
                raise ValueError(f"Invalid IANA timezone: {v}") from exc
        return v.strip()

    @model_validator(mode="after")
    def _normalize_title(self) -> "ScheduleInterviewRequest":
        t = (self.title or "").strip()
        if not t:
            role = (self.role or "").strip() or "Interview"
            self.title = f"Interview - {role}"
        return self


class ScheduleInterviewResponse(BaseModel):
    meeting_url: Optional[str] = Field(default=None, alias="meetingUrl")
    meeting_id: str = Field(alias="meetingId")
    status: str = "scheduled"

    model_config = {"populate_by_name": True}


class MeetingOut(BaseModel):
    id: str = Field(alias="_id")
    candidate_id: str
    email: EmailStr
    starts_at: str
    ends_at: str
    timezone: str
    duration_mins: int
    title: str
    notes: Optional[str] = None
    status: str
    meeting_url: Optional[str] = None
    created_at: str

    model_config = {"populate_by_name": True}


class EligibleCandidate(BaseModel):
    _id: str
    name: Optional[str] = None
    email: Optional[str] = None
    job_role: Optional[str] = None
    test_score: Optional[float] = None
    avatar: Optional[str] = None


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))


def _meeting_secret() -> str:
    return os.getenv("MEETING_ACCESS_SECRET") or os.getenv("JWT_SECRET") or "unsafe-dev-secret"


def _make_candidate_access_token(meeting_token: str, candidate_id: str, email: str, expires_at: datetime) -> str:
    payload = {
        "mt": meeting_token,
        "cid": candidate_id,
        "em": str(email or "").strip().lower(),
        "exp": int(expires_at.astimezone(timezone.utc).timestamp()),
    }
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig_b64 = _b64url(hmac.new(_meeting_secret().encode(), payload_b64.encode(), hashlib.sha256).digest())
    return f"{payload_b64}.{sig_b64}"


def _verify_candidate_access_token(token: str, meeting_token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        expected = _b64url(hmac.new(_meeting_secret().encode(), payload_b64.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig_b64):
            raise HTTPException(status_code=403, detail="Invalid meeting access token")
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if payload.get("mt") != meeting_token:
            raise HTTPException(status_code=403, detail="Meeting access token mismatch")
        if int(payload.get("exp") or 0) <= int(datetime.now(timezone.utc).timestamp()):
            raise HTTPException(status_code=403, detail="Meeting access token expired")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid meeting access token")


def _to_aware_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    raw = serialize_utc_any(value)
    return datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else None


def _meeting_window(doc: Dict[str, Any]) -> tuple[datetime, datetime]:
    starts = _to_aware_utc(doc.get("starts_at"))
    if not starts:
        raise HTTPException(status_code=500, detail="Meeting has invalid starts_at")
    ends = _to_aware_utc(doc.get("ends_at")) or (starts + timedelta(minutes=int(doc.get("duration_mins") or 45)))
    return starts, ends


def _meeting_state_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    starts_at, ends_at = _meeting_window(doc)
    now = datetime.now(timezone.utc)
    state = "countdown" if now < starts_at else ("expired" if now >= ends_at else "open")
    out = {
        "token": doc.get("token"),
        "title": doc.get("title"),
        "status": doc.get("status"),
        "candidate_id": doc.get("candidate_id"),
        "email": doc.get("email"),
        "timezone": doc.get("timezone") or "UTC",
        "duration_mins": int(doc.get("duration_mins") or 45),
        "starts_at": serialize_utc(starts_at),
        "ends_at": serialize_utc(ends_at),
        "now": serialize_utc(now),
        "state": state,
        "notes": doc.get("notes"),
    }
    external = str(doc.get("google_meet_url") or doc.get("external_url") or "").strip()
    if state == "open" and external:
        out["join_url"] = external
    return out


def _email_of(doc: Dict[str, Any]) -> str:
    return ((doc.get("email") or "") or ((doc.get("resume") or {}).get("email") or "")).strip()


def _job_role_of(doc: Dict[str, Any]) -> str:
    return doc.get("job_role") or doc.get("category") or doc.get("predicted_role") or (doc.get("resume") or {}).get("role") or "General"


def _format_local_dt(dt_utc: datetime, tz_name: str) -> str:
    try:
        if ZoneInfo:
            return dt_utc.astimezone(ZoneInfo(tz_name)).strftime("%a, %d %b %Y · %I:%M %p %Z")
    except Exception:
        pass
    return dt_utc.strftime("%a, %d %b %Y · %H:%M UTC")


def _invite_html(candidate_name: Optional[str], title: str, local_datetime_str: str, meeting_url: str, notes: Optional[str], timezone_name: str, duration_mins: int) -> str:
    who = candidate_name or "Candidate"
    safe_notes = f"<p><strong>Notes:</strong> {notes}</p>" if notes else ""
    return f"""
<p>Hi <strong>{who}</strong>,</p>
<p>You are invited to an interview.</p>
<p><strong>Title:</strong> {title}<br/>
<strong>When:</strong> {local_datetime_str} ({timezone_name})<br/>
<strong>Duration:</strong> {duration_mins} minutes</p>
{safe_notes}
<p>Join using this link:<br/>
<a href="{meeting_url}">{meeting_url}</a></p>
<p>Best,<br/>SmartHirex Team</p>
""".strip()


async def _assert_eligible(candidate_id: str, owner_id: str) -> Dict[str, Any]:
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id}, {"_id": 1, "name": 1, "email": 1, "resume.email": 1})
    if not candidate:
        raise HTTPException(status_code=403, detail="Candidate is not eligible for meeting scheduling.")
    submission = await db.test_submissions.find_one({"candidateId": candidate_id}, {"_id": 1, "testId": 1})
    if not submission:
        raise HTTPException(status_code=403, detail="Candidate is not eligible for meeting scheduling.")
    test_filter: Dict[str, Any] = {"candidateId": candidate_id, "status": "submitted"}
    if submission.get("testId"):
        test_filter = {"_id": submission.get("testId"), "candidateId": candidate_id, "status": "submitted"}
    submitted_test = await db.tests.find_one(test_filter, {"_id": 1}) or await db.tests.find_one({"candidateId": candidate_id, "status": "submitted"}, {"_id": 1})
    if not submitted_test:
        raise HTTPException(status_code=403, detail="Candidate is not eligible for meeting scheduling.")
    return candidate


async def _get_recruiter_if_any(request: Request) -> Optional[Any]:
    try:
        return await get_current_user(request)
    except Exception:
        return None


@router.post("/schedule", response_model=ScheduleInterviewResponse, status_code=status.HTTP_201_CREATED)
async def schedule_interview(req: ScheduleInterviewRequest, current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", ""))
    candidate = await _assert_eligible(req.candidate_id, owner_id)
    now = datetime.now(timezone.utc)
    if req.starts_at <= now:
        raise HTTPException(status_code=400, detail="startsAt must be in the future.")

    starts_at = req.starts_at.astimezone(timezone.utc)
    ends_at = starts_at + timedelta(minutes=req.duration_mins)
    token = uuid.uuid4().hex
    external_url = os.getenv("GOOGLE_MEET_URL_TEMPLATE", "https://meet.google.com/lookup/{token}").replace("{token}", token)
    access = _make_candidate_access_token(token, req.candidate_id, str(req.email), ends_at + timedelta(minutes=10))
    meeting_url = f"{build_meeting_url(token)}?access={quote(access)}"

    doc = {
        "candidate_id": req.candidate_id,
        "ownerUserId": owner_id,
        "email": str(req.email),
        "starts_at": starts_at,
        "ends_at": ends_at,
        "timezone": req.timezone,
        "duration_mins": req.duration_mins,
        "title": req.title,
        "notes": req.notes,
        "status": "scheduled",
        "token": token,
        "meeting_url": meeting_url,
        "google_meet_url": external_url,
        "external_url": external_url,
        "created_at": now,
        "email_sent": False,
    }
    try:
        res = await db.meetings.insert_one(doc)
        meeting_id = str(res.inserted_id)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DB insert failed: {exc}")

    dry_run = os.getenv("INVITES_DRY_RUN", "0").lower() in {"1", "true", "yes", "on"}
    subject = (req.title or "Interview Invitation").strip() or "Interview Invitation"
    html_body = _invite_html(req.candidate_name or candidate.get("name"), subject, _format_local_dt(starts_at, req.timezone), meeting_url, req.notes, req.timezone, req.duration_mins)
    try:
        if not dry_run:
            if not _send_interview_invite:
                raise RuntimeError("Email utility not available. Implement app.utils.emailer.send_interview_invite().")
            _send_interview_invite(to=str(req.email), subject=subject, html=html_body, meta={"candidate_id": req.candidate_id, "meeting_id": meeting_id, "starts_at_utc": serialize_utc(starts_at), "ends_at_utc": serialize_utc(ends_at)})
            await db.meetings.update_one({"_id": res.inserted_id}, {"$set": {"email_sent": True, "email_sent_at": datetime.now(timezone.utc)}})
        else:
            await db.meetings.update_one({"_id": res.inserted_id}, {"$set": {"email_sent": False, "email_error": "DRY_RUN"}})
    except Exception as exc:
        await db.meetings.update_one({"_id": res.inserted_id}, {"$set": {"email_sent": False, "email_error": str(exc), "status": "scheduled_email_failed"}})
        raise HTTPException(status_code=500, detail=f"Failed to send invite email: {exc}")

    return ScheduleInterviewResponse(meeting_url=meeting_url, meeting_id=meeting_id, status="scheduled")


@router.get("/by-candidate/{candidate_id}", response_model=List[MeetingOut])
async def list_interviews_for_candidate(candidate_id: str, current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", ""))
    if not await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id}, {"_id": 1}):
        raise HTTPException(status_code=403, detail="Not allowed to access this candidate meetings.")
    rows = await db.meetings.find({"candidate_id": candidate_id, "ownerUserId": owner_id}).sort("starts_at", -1).to_list(length=200)
    out: List[MeetingOut] = []
    for m in rows:
        starts = _to_aware_utc(m.get("starts_at")) or datetime.now(timezone.utc)
        ends = _to_aware_utc(m.get("ends_at")) or (starts + timedelta(minutes=int(m.get("duration_mins") or 45)))
        out.append(MeetingOut(_id=str(m.get("_id")), candidate_id=str(m.get("candidate_id")), email=str(m.get("email")), starts_at=serialize_utc(starts), ends_at=serialize_utc(ends), timezone=str(m.get("timezone") or "UTC"), duration_mins=int(m.get("duration_mins") or 45), title=str(m.get("title") or "Interview"), notes=m.get("notes"), status=str(m.get("status") or "scheduled"), meeting_url=m.get("meeting_url"), created_at=serialize_utc_any(m.get("created_at")) or serialize_utc(datetime.now(timezone.utc))))
    return out


@router.get("/eligible", response_model=List[EligibleCandidate])
async def list_eligible_candidates(limit: int = Query(100, ge=1, le=500), current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", ""))
    submitted_ids = await db.tests.distinct("candidateId", {"ownerUserId": owner_id, "status": "submitted"})
    docs = await db.parsed_resumes.find({"ownerUserId": owner_id, "_id": {"$in": submitted_ids or []}}, projection={"_id": 1, "name": 1, "email": 1, "resume.email": 1, "avatar": 1, "job_role": 1, "category": 1, "predicted_role": 1, "test_score": 1}).sort("updatedAt", -1).limit(limit).to_list(length=limit)
    return [EligibleCandidate(_id=str(c.get("_id")), name=c.get("name"), email=_email_of(c), job_role=_job_role_of(c), test_score=c.get("test_score"), avatar=c.get("avatar")) for c in docs]


@router.get("/stats/today")
async def stats_today(tz: str = Query("UTC"), current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", ""))
    zone = timezone.utc
    if ZoneInfo:
        try:
            zone = ZoneInfo(tz)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {tz}") from exc
    now_local = datetime.now(timezone.utc).astimezone(zone)
    start_utc = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    end_utc = (now_local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(timezone.utc)
    count = await db.meetings.count_documents({"ownerUserId": owner_id, "starts_at": {"$gte": start_utc, "$lt": end_utc}, "status": {"$in": ["scheduled", "rescheduled"]}})
    return {"scheduled_today": int(count), "timezone": tz}


@router.get("/by-token/{token}")
async def get_meeting_by_token_for_recruiter(token: str, current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", ""))
    doc = await db.meetings.find_one({"token": token, "ownerUserId": owner_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Meeting not found")
    payload = _meeting_state_payload(doc)
    payload["meeting_url"] = doc.get("meeting_url")
    return payload


@router.get("/access/{token}")
async def access_meeting(token: str, request: Request, access: Optional[str] = Query(default=None)):
    doc = await db.meetings.find_one({"token": token})
    if not doc:
        raise HTTPException(status_code=404, detail="Meeting not found")
    recruiter = await _get_recruiter_if_any(request)
    if recruiter:
        if str(doc.get("ownerUserId") or "") != str(getattr(recruiter, "id", "")):
            raise HTTPException(status_code=403, detail="Not authorized to access this meeting.")
    else:
        if not access:
            raise HTTPException(status_code=403, detail="Meeting access token required.")
        payload = _verify_candidate_access_token(access, token)
        if payload.get("cid") != str(doc.get("candidate_id")):
            raise HTTPException(status_code=403, detail="Not authorized to access this meeting.")
        token_email = str(payload.get("em") or "").strip().lower()
        doc_email = str(doc.get("email") or "").strip().lower()
        if token_email and doc_email and token_email != doc_email:
            raise HTTPException(status_code=403, detail="Not authorized to access this meeting.")
    return _meeting_state_payload(doc)
