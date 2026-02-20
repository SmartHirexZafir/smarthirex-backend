# app/routers/tests_router.py
from __future__ import annotations

import os
import re
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Literal

from fastapi import APIRouter, HTTPException, Path, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.utils.mongo import db  # AsyncIOMotorDatabase
from app.utils.emailer import send_email, render_invite_html
from app.models.auto_test_models import Candidate
from app.logic.generator import generate_test
from app.logic.evaluator import evaluate_test  # returns {"total_score": int, "details": [...]}
from app.routers.auth_router import get_current_user  # ✅ auth dependency
from app.utils.datetime_serialization import serialize_utc

# --- Optional graders (robust import; fully optional) ------------------------
USE_RULES = os.getenv("FREEFORM_RULES_GRADING", "1").strip().lower() in {"1", "true", "yes", "on"}
USE_LLM = os.getenv("FREEFORM_LLM_GRADING", "0").strip().lower() in {"1", "true", "yes", "on"}
FREEFORM_MAX_POINTS = int(os.getenv("FREEFORM_MAX_POINTS", "5") or "5")

_rules_grader = None
_llm_grader = None

if USE_RULES:
    try:
        from app.logic.grader_rules import grade_free_form as _rules_grade  # type: ignore

        _rules_grader = _rules_grade
    except Exception:
        _rules_grader = None
        USE_RULES = False  # disable gracefully

if USE_LLM:
    try:
        from app.logic.grader_llm import grade_free_form_llm as _llm_grade  # type: ignore

        _llm_grader = _llm_grade
    except Exception:
        _llm_grader = None
        USE_LLM = False  # disable gracefully

# --- Optional code runner (robust import; fully optional) --------------------
# Enable with CODE_RUNNER=1 (or true/yes/on)
CODE_RUNNER_ENABLED = os.getenv("CODE_RUNNER", "0").strip().lower() in {"1", "true", "yes", "on"}
# Time/memory defaults (can be overridden per question)
RUNNER_TIME_LIMIT = int(os.getenv("RUNNER_TIME_LIMIT_SEC", "2") or "2")
RUNNER_MEM_MB = int(os.getenv("RUNNER_MEMORY_MB", "128") or "128")

_runner_run = None
if CODE_RUNNER_ENABLED:
    # Try multiple likely import paths, but stay graceful if unavailable
    try:
        from app.services.code_runner import run_submission as _runner_run  # type: ignore
    except Exception:
        try:
            from services.code_runner import run_submission as _runner_run  # type: ignore
        except Exception:
            _runner_run = None
            CODE_RUNNER_ENABLED = False
# ---------------------------------------------------------------------------

router = APIRouter()
logger = logging.getLogger(__name__)

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
TEST_TOKEN_EXPIRY_MINUTES = int(os.getenv("TEST_TOKEN_EXPIRY_MINUTES", "60"))

# ---------- helpers ----------

# ✅ Helper: Safe UTC datetime parsing from strings or objects
def _parse_utc_datetime(dt_input: Optional[str | datetime]) -> Optional[datetime]:
    """
    Parse a datetime string or object to UTC-naive datetime.
    Handles ISO strings, ISO with Z suffix, and datetime objects.
    Always returns UTC-naive datetime (or None if invalid).
    """
    if not dt_input:
        return None
    
    if isinstance(dt_input, datetime):
        # Already a datetime object
        if dt_input.tzinfo is not None:
            from datetime import timezone
            # Convert to UTC and remove tzinfo
            return dt_input.astimezone(timezone.utc).replace(tzinfo=None)
        return dt_input
    
    # String parsing
    try:
        from datetime import timezone
        dt_str = str(dt_input).strip()
        
        # Handle 'Z' suffix (UTC)
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1] + '+00:00'
        
        # Try parsing
        dt = datetime.fromisoformat(dt_str)
        
        # Convert to UTC-naive if timezone-aware
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        
        return dt
    except (ValueError, AttributeError, TypeError):
        return None


def _extract_int_years(value: Optional[str]) -> int:
    if not value:
        return 0
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 0


def _safe_experience_years(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return max(0.0, float(value))
        except Exception:
            return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return 0.0
    try:
        return max(0.0, float(m.group(1)))
    except Exception:
        return 0.0


def _experience_years_from_doc(doc: dict) -> float:
    for key in ("total_experience_years", "years_of_experience", "experience_years", "yoe", "experience"):
        years = _safe_experience_years(doc.get(key))
        if years > 0:
            return years
    resume = doc.get("resume", {}) if isinstance(doc.get("resume"), dict) else {}
    for key in ("experience", "totalExperience"):
        years = _safe_experience_years(resume.get(key))
        if years > 0:
            return years
    return 0.0


def _best_recruiter_name(user_doc: Optional[Dict[str, Any]], fallback_email: str = "") -> str:
    if not isinstance(user_doc, dict):
        return fallback_email or "Recruiter"
    for k in ("name", "full_name", "display_name"):
        v = _norm_text(user_doc.get(k))
        if v:
            return v
    first = _norm_text(user_doc.get("first_name") or user_doc.get("firstName"))
    last = _norm_text(user_doc.get("last_name") or user_doc.get("lastName"))
    full = f"{first} {last}".strip()
    if full:
        return full
    email = _norm_text(user_doc.get("email")) or fallback_email
    return email or "Recruiter"


def _role_from_candidate_doc(doc: dict) -> str:
    return (
        doc.get("predicted_role")
        or doc.get("category")
        or doc.get("resume", {}).get("role")
        or "General"
    )


def _email_from_candidate_doc(doc: dict) -> Optional[str]:
    return (
        doc.get("email")
        or doc.get("resume", {}).get("email")
        or doc.get("contact", {}).get("email")
    )


def _name_from_candidate_doc(doc: dict) -> str:
    return (
        doc.get("name")
        or doc.get("resume", {}).get("name")
        or doc.get("profile", {}).get("name")
        or "Candidate"
    )


def _skills_from_candidate_doc(doc: dict) -> List[str]:
    return doc.get("skills") or doc.get("resume", {}).get("skills") or []


def _experience_string_from_doc(doc: dict) -> str:
    years = _experience_years_from_doc(doc)
    return f"{years:g} years"


def _sanitize_question_count(n: Optional[int]) -> int:
    """
    Clamp to a safe range; default to 4 if not provided.
    """
    try:
        v = int(n or 4)
    except Exception:
        v = 4
    # sensible bounds so users can’t request 1000 questions
    return max(1, min(50, v))


def _norm_text(v: Any) -> str:
    try:
        return str(v or "").strip()
    except Exception:
        return ""


def _to_utc_iso_z(dt_input: Optional[str | datetime]) -> Optional[str]:
    dt = _parse_utc_datetime(dt_input)
    if not dt:
        return None
    return serialize_utc(dt)


def _one_test_conflict(message: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"error": "conflict", "message": message, "code": "ONE_TEST_CONFLICT"},
    )


def _question_fingerprint(text: Any) -> str:
    t = _norm_text(text).lower()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _question_set_fingerprint(questions: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        parts.append(_question_fingerprint(q.get("question", "")))
    payload = "|".join([p for p in parts if p])
    if not payload:
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _profile_signature(role: Any, experience_years: Any) -> str:
    role_norm = _norm_text(role).lower() or "general"
    try:
        years = float(experience_years or 0.0)
    except Exception:
        years = 0.0
    # Bucket to reduce fragile precision mismatches while keeping "same experience" intent.
    years_bucket = int(round(max(0.0, years)))
    return f"{role_norm}|{years_bucket}"


async def _candidate_has_any_attempt(candidate_id: str) -> bool:
    """
    True if candidate already has a submitted test attempt.
    """
    existing_submission = await db.test_submissions.find_one({"candidateId": candidate_id}, {"_id": 1})
    return bool(existing_submission)


def _answer_has_content(ans: Dict[str, Any]) -> bool:
    return bool(_norm_text(ans.get("answer", "")))


def _normalize_answers(raw_answers: Any, total_questions: int) -> List[Dict[str, Any]]:
    safe: List[Dict[str, Any]] = []
    src = raw_answers if isinstance(raw_answers, list) else []
    for i in range(max(0, int(total_questions))):
        item = src[i] if i < len(src) and isinstance(src[i], dict) else {}
        safe.append(
            {
                "answer": _norm_text(item.get("answer", "")),
                "type": item.get("type"),
                "language": item.get("language"),
                "question_id": item.get("question_id"),
            }
        )
    return safe


def _merge_answers(
    preferred: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
    total_questions: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(max(0, int(total_questions))):
        a = preferred[i] if i < len(preferred) else {"answer": ""}
        b = fallback[i] if i < len(fallback) else {"answer": ""}
        out.append(a if _answer_has_content(a) else b)
    return out


def _code_tests_from_question(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize question-defined tests to the runner schema if present.
    Expected keys (best-effort): name, input, args, expected_stdout, match
    If not present, return [] (runner will still compile/run w/o tests).
    """
    tests = q.get("tests")
    if not isinstance(tests, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "name": _norm_text(t.get("name") or f"t{i+1}"),
                "input": _norm_text(t.get("input") or ""),
                "args": t.get("args") if isinstance(t.get("args"), list) else [],
                "expected_stdout": t.get("expected_stdout"),
                "match": _norm_text(t.get("match") or "exact") or "exact",
            }
        )
    return out


# ---------- request/response models ----------

TestType = Literal["smart", "custom"]


class CustomQuestion(BaseModel):
    question: str = Field(..., description="Question prompt / body text")
    type: Optional[str] = Field(default="text", description="mcq | code | scenario | text | freeform")
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    tests: Optional[List[Dict[str, Any]]] = None  # for code
    language: Optional[str] = None
    id: Optional[Any] = None


class CustomTest(BaseModel):
    title: Optional[str] = None
    questions: List[Dict[str, Any]] | List[CustomQuestion] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def _non_empty(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("custom.questions must be a non-empty list")
        return v


class TestComposition(BaseModel):
    """Composition parameters for Smart AI Test - sender decides question types"""
    mcq_count: int = Field(default=0, ge=0, le=50, description="Number of MCQ questions")
    scenario_count: int = Field(default=0, ge=0, le=50, description="Number of scenario questions")
    code_count: Optional[int] = Field(default=0, ge=0, le=50, description="Number of coding questions (optional)")


class InviteRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate _id from parsed_resumes")
    subject: Optional[str] = "Your SmartHirex Assessment"
    body_html: Optional[str] = None  # optional custom HTML; {TEST_LINK} will be replaced
    # allow sender to choose number of questions (for custom tests or backward compatibility)
    question_count: Optional[int] = Field(default=4, ge=1, le=50)
    # NEW: test type & optional custom
    test_type: Optional[TestType] = Field(default="smart")
    custom: Optional[CustomTest] = None
    # NEW: composition for Smart AI Test (sender decides question types)
    composition: Optional[TestComposition] = None
    # ✅ NEW: Scheduled test timing
    scheduled_date_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime string for when the test should become available (UTC). If not provided, test is available immediately."
    )
    # ✅ NEW: Test duration in minutes
    test_duration_minutes: Optional[int] = Field(
        default=60,
        ge=5,
        le=300,
        description="Test duration in minutes (5-300). Test will auto-submit when duration expires."
    )


class InviteResponse(BaseModel):
    invite_id: str
    token: str
    test_link: str
    email: str
    sent: bool
    expires_at: str


class StartTestRequest(BaseModel):
    token: str


class StartTestResponse(BaseModel):
    test_id: str
    candidate_id: str
    questions: List[Dict[str, Any]]  # raw dicts for full compatibility
    scheduled_datetime: Optional[str] = None  # ✅ For countdown display
    expires_at: Optional[str] = None  # ✅ Test expiration time
    duration_minutes: Optional[int] = None  # ✅ Test duration


class SubmitTestRequest(BaseModel):
    token: str
    # answers must be like [{"answer": "..."}] – matches your evaluator
    # New optional fields per answer are tolerated and ignored by base evaluator:
    #   type?: "mcq" | "code" | "scenario"
    #   language?: string
    #   question_id?: str|int
    answers: List[Dict[str, Any]]


class SubmitTestResponse(BaseModel):
    test_id: str
    candidate_id: str
    score: float  # unified percent (0.0 for ungraded custom)
    details: List[Dict[str, Any]]


class AutosaveRequest(BaseModel):
    token: str
    answers: List[Dict[str, Any]]


class AutosaveResponse(BaseModel):
    ok: bool
    status: str
    saved_at: str
    auto_submitted: bool = False


class InviteEligibilityResponse(BaseModel):
    candidate_id: str
    can_invite: bool
    reason: Optional[str] = None
    source: Optional[str] = None


# ---------- routes ----------


@router.post("/invite", response_model=InviteResponse)
async def create_invite(req: InviteRequest, current=Depends(get_current_user)):
    """
    Create a test invite for a candidate and send an email with a secure link.

    ENHANCEMENTS:
    - Supports test_type: "smart" (default) or "custom"
    - When custom, persist provided questions & title on the invite
    - ENFORCES: A candidate can receive only ONE type of test (Smart AI OR Custom, not both)
    """
    candidate = await db.parsed_resumes.find_one({"_id": req.candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    email = _email_from_candidate_doc(candidate)
    if not email:
        raise HTTPException(status_code=400, detail="Candidate email not found")

    # Strict single-attempt policy: if already attempted, block new test creation.
    if await _candidate_has_any_attempt(req.candidate_id):
        raise _one_test_conflict("Candidate already has an attempted test. Only one test attempt is allowed (Smart AI or Custom).")

    # ✅ ENFORCE: Check if candidate already has a test assigned (pending/active)
    existing_invite = await db.test_invites.find_one({
        "candidateId": req.candidate_id,
        "status": {"$in": ["pending", "active"]}
    })
    if existing_invite:
        existing_type = existing_invite.get("type", "smart")
        raise _one_test_conflict(
            f"Candidate already has a {existing_type} test assigned. Only one test type (Smart AI or Custom) is allowed per candidate."
        )

    # ✅ Also check if candidate has a completed test
    existing_test = await db.tests.find_one({
        "candidateId": req.candidate_id,
        "status": {"$in": ["active", "submitted"]}
    })
    if existing_test:
        existing_test_type = existing_test.get("type", "smart")
        raise _one_test_conflict(
            f"Candidate already has a {existing_test_type} test. Only one test type (Smart AI or Custom) is allowed per candidate."
        )

    name = _name_from_candidate_doc(candidate)
    role = _role_from_candidate_doc(candidate)
    owner_user_id = str(getattr(current, "id", "") or "")
    recruiter_email = str(getattr(current, "email", "") or "")
    recruiter_doc = await db.users.find_one({"_id": owner_user_id}) if owner_user_id else None
    recruiter_name = _best_recruiter_name(recruiter_doc, recruiter_email)

    token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(minutes=TEST_TOKEN_EXPIRY_MINUTES)
    test_link = f"{FRONTEND_BASE_URL.rstrip('/')}/test/{token}"

    question_count = _sanitize_question_count(req.question_count)
    test_type: TestType = req.test_type or "smart"

    # persist custom payload when applicable
    custom_payload: Optional[Dict[str, Any]] = None
    composition_payload: Optional[Dict[str, Any]] = None
    
    if test_type == "custom":
        if not req.custom:
            raise HTTPException(status_code=400, detail="custom test requires 'custom' payload")
        # Store as plain dict for flexibility
        custom_payload = {
            "title": req.custom.title or "Custom Test",
            "questions": [
                q
                if isinstance(q, dict)
                else (q.model_dump() if hasattr(q, "model_dump") else q.dict())
                for q in req.custom.questions
            ],
        }
    elif test_type == "smart" and req.composition:
        # Store composition parameters for Smart AI Test
        composition_payload = {
            "mcq_count": req.composition.mcq_count,
            "scenario_count": req.composition.scenario_count,
            "code_count": req.composition.code_count or 0,
        }
        # Update question_count to match composition
        question_count = composition_payload["mcq_count"] + composition_payload["scenario_count"] + composition_payload["code_count"]
        if question_count == 0:
            raise HTTPException(status_code=400, detail="Smart AI Test must have at least one question (MCQ, scenario, or code)")

    # ✅ Parse scheduled_date_time from request using safe UTC parsing
    scheduled_datetime: Optional[datetime] = None
    if req.scheduled_date_time:
        scheduled_datetime = _parse_utc_datetime(req.scheduled_date_time)
    
    # ✅ Get test duration from request
    test_duration_minutes = req.test_duration_minutes or 60

    invite_doc = {
        "_id": uuid.uuid4().hex,
        "candidateId": candidate["_id"],
        "email": email,
        "name": name,
        "role": role,
        "experienceYears": int(_experience_years_from_doc(candidate)),
        "status": "pending",
        "token": token,
        "expiresAt": expires_at,
        "createdAt": datetime.utcnow(),
        "questionCount": question_count,
        "type": test_type,  # <-- NEW
        "custom": custom_payload,  # <-- NEW (optional)
        "composition": composition_payload,  # <-- NEW (optional, for Smart AI Test)
        "ownerUserId": owner_user_id or None,
        "invitedByUserId": owner_user_id or None,
        "recruiterName": recruiter_name,
        "recruiterEmail": recruiter_email or None,
        # ✅ NEW: Scheduled timing and duration
        "scheduledDateTime": _to_utc_iso_z(scheduled_datetime) if scheduled_datetime else None,
        "testDurationMinutes": test_duration_minutes,
    }
    try:
        await db.test_invites.insert_one(invite_doc)
    except DuplicateKeyError:
        raise _one_test_conflict("Candidate already has an active or completed test attempt. Cannot create another invite.")

    # ✅ Enhanced email template with scheduled time and duration
    html = req.body_html or render_invite_html(
        candidate_name=name,
        role=role,
        test_link=test_link,
        scheduled_datetime=scheduled_datetime,
        duration_minutes=test_duration_minutes,
    )
    html = html.replace("{TEST_LINK}", test_link).replace("{{TEST_LINK}}", test_link)
    if scheduled_datetime:
        html = html.replace("{SCHEDULED_TIME}", scheduled_datetime.strftime("%Y-%m-%d %H:%M UTC"))
    html = html.replace("{DURATION}", str(test_duration_minutes))

    send_email(to=email, subject=req.subject or "Your SmartHirex Assessment", html=html)

    return InviteResponse(
        invite_id=invite_doc["_id"],
        token=token,
        test_link=test_link,
        email=email,
        sent=True,
        expires_at=_to_utc_iso_z(expires_at) or serialize_utc(expires_at),
    )


@router.get("/eligibility/{candidate_id}", response_model=InviteEligibilityResponse)
async def get_invite_eligibility(candidate_id: str, current=Depends(get_current_user)):
    owner_id = str(getattr(current, "id", None) or "")
    cand = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id}, {"_id": 1})
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if await _candidate_has_any_attempt(candidate_id):
        return InviteEligibilityResponse(
            candidate_id=candidate_id,
            can_invite=False,
            reason="Candidate already has an attempted test.",
            source="test_submissions",
        )
    existing_invite = await db.test_invites.find_one(
        {"candidateId": candidate_id, "status": {"$in": ["pending", "active"]}},
        {"_id": 1},
    )
    if existing_invite:
        return InviteEligibilityResponse(
            candidate_id=candidate_id,
            can_invite=False,
            reason="Candidate already has an active invite.",
            source="test_invites",
        )
    existing_test = await db.tests.find_one(
        {"candidateId": candidate_id, "status": {"$in": ["active", "submitted"]}},
        {"_id": 1},
    )
    if existing_test:
        return InviteEligibilityResponse(
            candidate_id=candidate_id,
            can_invite=False,
            reason="Candidate already has an active/submitted test.",
            source="tests",
        )
    return InviteEligibilityResponse(candidate_id=candidate_id, can_invite=True, reason=None, source=None)


@router.post("/start", response_model=StartTestResponse)
async def start_test(req: StartTestRequest):
    """
    Validate token & expiry, then generate a test based on sender's composition choices.

    ENHANCEMENTS:
    - If invite.type == "custom", return the custom questions instead of generating.
    - If invite.type == "smart", use composition parameters (mcq_count, scenario_count, code_count)
      provided by the sender. Difficulty is still adapted from candidate experience.
    - ✅ NEW: Validates scheduled time - test cannot start before scheduled time
    - ✅ NEW: Validates 30-minute expiration after test starts
    """
    invite = await db.test_invites.find_one({"token": req.token})
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.get("status") not in {"pending", "active"}:
        raise HTTPException(status_code=400, detail=f"Invite status is {invite.get('status')}")
    
    now = datetime.utcnow()
    
    # ✅ Check scheduled time - test cannot start before scheduled time
    scheduled_dt_str = invite.get("scheduledDateTime")
    if scheduled_dt_str:
        scheduled_dt = _parse_utc_datetime(scheduled_dt_str)
        if scheduled_dt and now < scheduled_dt:
            # Test not yet available - return scheduled time for countdown
            scheduled_iso = _to_utc_iso_z(scheduled_dt) or serialize_utc(scheduled_dt)
            raise HTTPException(
                status_code=403,
                detail=f"Test is scheduled for {scheduled_iso}. Please wait until the scheduled time. SCHEDULED_DATETIME:{scheduled_iso}"
            )
    
    # ✅ Check if test has already started - if so, enforce actual test expiry
    existing_test = await db.tests.find_one({"token": req.token})
    if existing_test:
        expires_dt = _parse_utc_datetime(existing_test.get("expiresAt"))
        if expires_dt and now > expires_dt:
            if existing_test.get("status") != "submitted":
                try:
                    await submit_test(SubmitTestRequest(token=req.token, answers=[]), forced_by_expiry=True)
                except Exception as e:
                    logger.exception("forced-submit from /tests/start failed token=%s err=%s", req.token, e)
            raise HTTPException(
                status_code=410,
                detail=f"Test time expired at {_to_utc_iso_z(expires_dt) or serialize_utc(expires_dt)}. If not already submitted, it will be auto-submitted.",
            )
        # Test already started and still active - return existing persisted test
        expires_at = existing_test.get("expiresAt")
        if isinstance(expires_at, datetime):
            expires_at_str = _to_utc_iso_z(expires_at) or serialize_utc(expires_at)
        elif isinstance(expires_at, str):
            expires_at_str = _to_utc_iso_z(expires_at) or expires_at
        else:
            expires_at_str = None
        return StartTestResponse(
            test_id=existing_test["_id"],
            candidate_id=existing_test["candidateId"],
            questions=existing_test.get("questions", []),
            scheduled_datetime=_to_utc_iso_z(invite.get("scheduledDateTime")) or invite.get("scheduledDateTime"),
            expires_at=expires_at_str,
            duration_minutes=existing_test.get("testDurationMinutes"),
        )
    
    # ✅ Check general invite expiration
    invite_expires = invite.get("expiresAt")
    if isinstance(invite_expires, str):
        invite_expires = _parse_utc_datetime(invite_expires)
    
    if invite_expires and now > invite_expires:
        raise HTTPException(status_code=410, detail="Invite link expired")

    candidate = await db.parsed_resumes.find_one({"_id": invite["candidateId"]})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Prevent parallel/alternate-token starts for same candidate.
    other_test = await db.tests.find_one(
        {
            "candidateId": invite["candidateId"],
            "token": {"$ne": req.token},
            "status": {"$in": ["active", "submitted"]},
        },
        {"_id": 1, "type": 1, "status": 1},
    )
    if other_test:
        raise _one_test_conflict("Candidate already has a test in progress or submitted. Multiple tests are not allowed.")

    # API-level bypass guard: block starting a new test if candidate already attempted one.
    existing_submission = await db.test_submissions.find_one({"candidateId": invite["candidateId"]}, {"_id": 1, "testId": 1})
    if existing_submission:
        # Allow idempotent continuation only if submission belongs to this token's test.
        current_test = await db.tests.find_one({"token": req.token}, {"_id": 1})
        current_test_id = str((current_test or {}).get("_id") or "")
        submitted_test_id = str(existing_submission.get("testId") or "")
        if not current_test_id or submitted_test_id != current_test_id:
            raise _one_test_conflict("Candidate has already attempted a test. Multiple attempts are not allowed.")

    # Compute questions based on type
    test_type: TestType = invite.get("type", "smart")
    questions: List[Dict[str, Any]] = []
    question_count: int = 4  # Initialize with default value

    if test_type == "custom":
        custom = invite.get("custom") or {}
        questions = custom.get("questions") or []
        if not isinstance(questions, list) or not questions:
            # Fallback gracefully to generator if custom is empty/malformed
            test_type = "smart"
            questions = []  # Reset for smart path

    if test_type == "smart":
        role_value = _role_from_candidate_doc(candidate)
        experience_years = _experience_years_from_doc(candidate)
        profile_sig = _profile_signature(role_value, experience_years)
        candidate_model = Candidate(
            _id=candidate["_id"],
            name=_name_from_candidate_doc(candidate),
            email=_email_from_candidate_doc(candidate) or "",
            phone=None,
            skills=_skills_from_candidate_doc(candidate),
            experience=f"{experience_years:g} years",
            resume_text=candidate.get("raw_text", ""),
            job_role=role_value,
        )
        
        # Fetch historical question sets scoped to same role+experience profile.
        previous_questions: List[str] = []
        previous_q_fingerprints: set[str] = set()
        previous_set_fingerprints: set[str] = set()

        previous_tests = await db.tests.find(
            {"candidateId": candidate["_id"]},
            {"questions.question": 1, "questionSetFingerprint": 1, "profileSignature": 1},
        ).sort("startedAt", -1).limit(40).to_list(length=40)
        for prev_test in previous_tests:
            if prev_test.get("profileSignature") and prev_test.get("profileSignature") != profile_sig:
                continue
            prev_questions = prev_test.get("questions", [])
            if isinstance(prev_questions, list):
                for q in prev_questions:
                    q_text = _norm_text((q or {}).get("question", ""))
                    q_fp = _question_fingerprint(q_text)
                    if q_fp and q_fp not in previous_q_fingerprints:
                        previous_q_fingerprints.add(q_fp)
                        previous_questions.append(q_text)
                set_fp = prev_test.get("questionSetFingerprint") or _question_set_fingerprint(prev_questions)
                if set_fp:
                    previous_set_fingerprints.add(str(set_fp))
        previous_subs = await db.test_submissions.find(
            {"candidateId": candidate["_id"]},
            {"questions.question": 1, "questionSetFingerprint": 1, "profileSignature": 1},
        ).sort("submittedAt", -1).limit(40).to_list(length=40)
        for prev_sub in previous_subs:
            if prev_sub.get("profileSignature") and prev_sub.get("profileSignature") != profile_sig:
                continue
            prev_questions = prev_sub.get("questions", [])
            if isinstance(prev_questions, list):
                for q in prev_questions:
                    q_text = _norm_text((q or {}).get("question", ""))
                    q_fp = _question_fingerprint(q_text)
                    if q_fp and q_fp not in previous_q_fingerprints:
                        previous_q_fingerprints.add(q_fp)
                        previous_questions.append(q_text)
                set_fp = prev_sub.get("questionSetFingerprint") or _question_set_fingerprint(prev_questions)
                if set_fp:
                    previous_set_fingerprints.add(str(set_fp))
        
        # Use composition parameters if provided, otherwise fallback to question_count
        composition = invite.get("composition")
        seed_base = uuid.uuid4().hex
        selected_seed = f"{seed_base}:0"
        if composition:
            for retry in range(12):
                selected_seed = f"{seed_base}:{retry}"
                questions = generate_test(
                    candidate_model,
                    mcq_count=composition.get("mcq_count", 0),
                    scenario_count=composition.get("scenario_count", 0),
                    code_count=composition.get("code_count", 0),
                    previous_questions=previous_questions if previous_questions else None,
                    seed=selected_seed,
                )
                current_set_fp = _question_set_fingerprint(questions)
                if not current_set_fp or current_set_fp not in previous_set_fingerprints:
                    break
            # Calculate question_count from composition
            question_count = composition.get("mcq_count", 0) + composition.get("scenario_count", 0) + composition.get("code_count", 0)
            # Fallback to actual questions length if calculation fails
            if question_count == 0:
                question_count = len(questions) if questions else 4
        else:
            # Fallback for backward compatibility
            question_count = _sanitize_question_count(invite.get("questionCount", 4))
            for retry in range(12):
                selected_seed = f"{seed_base}:{retry}"
                questions = generate_test(
                    candidate_model,
                    question_count=question_count,
                    previous_questions=previous_questions if previous_questions else None,
                    seed=selected_seed,
                )
                current_set_fp = _question_set_fingerprint(questions)
                if not current_set_fp or current_set_fp not in previous_set_fingerprints:
                    break
    else:
        # custom path already set questions; still capture a sensible count
        question_count = len(questions) if questions else _sanitize_question_count(invite.get("questionCount", 4))
        profile_sig = _profile_signature(invite.get("role"), invite.get("experienceYears"))
        selected_seed = None

    if not isinstance(questions, list) or len(questions) == 0:
        # Never send an empty test attempt to candidate.
        raise HTTPException(
            status_code=502,
            detail="Unable to generate test questions at this time. Please try again shortly.",
        )

    # ✅ Get test duration from invite
    test_duration_minutes = invite.get("testDurationMinutes", 60)
    started_at = datetime.utcnow()
    expires_at = started_at + timedelta(minutes=test_duration_minutes)
    
    test_doc = {
        "_id": uuid.uuid4().hex,
        "inviteId": invite["_id"],
        "candidateId": candidate["_id"],
        "token": req.token,
        "questions": questions,
        "status": "active",
        "startedAt": started_at,
        "expiresAt": expires_at,  # ✅ Auto-submit time
        "questionCount": question_count,
        "type": test_type,  # <-- NEW
        "custom": invite.get("custom"),
        "composition": invite.get("composition"),
        "profileSignature": profile_sig,
        "questionSeed": selected_seed,
        "questionSetFingerprint": _question_set_fingerprint(questions),
        "testDurationMinutes": test_duration_minutes,  # ✅ Store duration
        "ownerUserId": invite.get("ownerUserId"),
        "invitedByUserId": invite.get("invitedByUserId"),
        "recruiterName": invite.get("recruiterName"),
        "recruiterEmail": invite.get("recruiterEmail"),
    }
    # Atomic upsert: prevents duplicate test documents on concurrent /start calls.
    persisted = await db.tests.find_one_and_update(
        {"token": req.token},
        {"$setOnInsert": test_doc},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if not persisted:
        raise HTTPException(status_code=500, detail="Failed to initialize test")

    if invite.get("status") == "pending":
        await db.test_invites.update_one({"_id": invite["_id"]}, {"$set": {"status": "active"}})

    return StartTestResponse(
        test_id=persisted["_id"],
        candidate_id=persisted["candidateId"],
        questions=persisted.get("questions", questions),
        scheduled_datetime=_to_utc_iso_z(invite.get("scheduledDateTime")) or invite.get("scheduledDateTime"),
        expires_at=(
            _to_utc_iso_z(persisted.get("expiresAt"))
            or (
                serialize_utc(persisted.get("expiresAt"))
                if isinstance(persisted.get("expiresAt"), datetime)
                else (_to_utc_iso_z(str(persisted.get("expiresAt") or "")) or str(persisted.get("expiresAt") or (_to_utc_iso_z(expires_at) or serialize_utc(expires_at))))
            )
        ),
        duration_minutes=persisted.get("testDurationMinutes", test_duration_minutes),
    )


@router.post("/submit", response_model=SubmitTestResponse)
async def submit_test(req: SubmitTestRequest, forced_by_expiry: bool = False):
    """
    Persist answers, grade with evaluator, return score + details.
    Also updates the candidate's 'test_score' (%) in parsed_resumes.

    ✅ ENFORCES: Test must not be expired (server-side validation using UTC time)

    Unchanged behavior for SMART tests:
    - MCQ percent is calculated exactly as before.
    - If the test was already submitted, the stored result is returned.

    Extensions:
    - CUSTOM tests are stored as 'needs_marking' and return score=0.0 until graded via /tests/grade/{attempt_id}.
    - Free-form/scenario items can be auto-graded via rules/LLM when enabled (SMART tests only).
    - Code items can be executed via an optional runner (SMART tests only); per-test results
      are attached to the corresponding detail row.
    """
    test = await db.tests.find_one({"token": req.token})
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    # Inspect invite for type
    invite = await db.test_invites.find_one({"token": req.token}) or {}
    test_type: TestType = test.get("type") or invite.get("type") or "smart"

    # If already submitted, return the stored result (idempotent)
    if test.get("status") == "submitted":
        existing = await db.test_submissions.find_one({"testId": test["_id"]})
        if existing:
            # Only propagate to candidate profile for graded (i.e., not needs_marking)
            if not existing.get("needs_marking", False):
                try:
                    await db.parsed_resumes.update_one(
                        {"_id": test["candidateId"]},
                        {"$set": {"test_score": float(existing.get("score", 0.0))}},
                    )
                except Exception:
                    pass
            return SubmitTestResponse(
                test_id=test["_id"],
                candidate_id=test["candidateId"],
                score=float(existing.get("score", 0.0)),
                details=existing.get("details", []),
            )
        # Defensive guard: once status is submitted, never re-finalize.
        if forced_by_expiry:
            logger.info("forced-submit skipped: already submitted token=%s testId=%s", req.token, test.get("_id"))
        raise _one_test_conflict("Test already submitted")

    questions: List[Dict[str, Any]] = test.get("questions", [])
    total_questions = len(questions)
    incoming_answers = _normalize_answers(req.answers, total_questions)
    saved_draft_answers = _normalize_answers(test.get("draftAnswers", []), total_questions)

    # Normal path: merge current payload with server draft to avoid answer loss.
    effective_answers = _merge_answers(incoming_answers, saved_draft_answers, total_questions)

    # Server-enforced timer validation (anti-devtools/API-time manipulation).
    now = datetime.utcnow()
    expires_dt = _parse_utc_datetime(test.get("expiresAt"))
    effective_forced_by_expiry = bool(forced_by_expiry or (expires_dt and now > expires_dt))
    if effective_forced_by_expiry:
        # After deadline, ignore new payload edits and lock to last server-saved draft.
        # If no draft exists, we fallback to merged answers to avoid empty-loss edge cases.
        effective_answers = saved_draft_answers if any(_answer_has_content(a) for a in saved_draft_answers) else effective_answers
        logger.info(
            "forced-submit executing token=%s testId=%s candidateId=%s draftCount=%s",
            req.token,
            test.get("_id"),
            test.get("candidateId"),
            len(saved_draft_answers),
        )

    # ---------- CUSTOM TEST PATH (manual marking) ----------
    if test_type == "custom":
        # Build lightweight details pairing question/answer for UI preview
        details: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            details.append(
                {
                    "question": q.get("question", ""),
                    "type": q.get("type", "text"),
                    "answer": (effective_answers[i].get("answer") if i < len(effective_answers) else None),
                    "is_correct": None,
                }
            )

        # ✅ Save COMPLETE test data for custom tests
        submission_doc = {
            "_id": uuid.uuid4().hex,
            "testId": test["_id"],
            "candidateId": test["candidateId"],
            "answers": effective_answers,  # Candidate's submitted answers
            "questions": questions,  # ✅ Complete questions from the custom test
            "score": 0.0,  # not graded yet
            "details": details,  # ✅ Complete details (question, answer, type, etc.)
            "submittedAt": datetime.utcnow(),
            "needs_marking": True,
            "type": "custom",
            # ✅ Include test metadata
            "testType": "custom",
            "custom": test.get("custom"),  # ✅ Include custom test title and metadata
            "questionCount": len(questions),
            "questionSeed": test.get("questionSeed"),
            "profileSignature": test.get("profileSignature"),
            "questionSetFingerprint": test.get("questionSetFingerprint") or _question_set_fingerprint(questions),
            "auto_submitted": effective_forced_by_expiry,
            "submit_reason": "expired_force_submit" if effective_forced_by_expiry else "manual_submit",
            "durationMinutes": test.get("testDurationMinutes") or invite.get("testDurationMinutes"),
            "ownerUserId": test.get("ownerUserId") or invite.get("ownerUserId"),
            "invitedByUserId": test.get("invitedByUserId") or invite.get("invitedByUserId"),
            "recruiterName": test.get("recruiterName") or invite.get("recruiterName"),
            "recruiterEmail": test.get("recruiterEmail") or invite.get("recruiterEmail"),
        }
        try:
            await db.test_submissions.insert_one(submission_doc)
        except DuplicateKeyError:
            raise _one_test_conflict("Duplicate submission detected for this test.")
        await db.tests.update_one({"_id": test["_id"]}, {"$set": {"status": "submitted"}})
        await db.test_invites.update_one({"token": req.token}, {"$set": {"status": "completed"}})

        # Note: we DO NOT update candidate.test_score for custom until graded
        return SubmitTestResponse(
            test_id=test["_id"],
            candidate_id=test["candidateId"],
            score=0.0,
            details=details,
        )

    # ---------- SMART TEST PATH (original logic + optional enhancers) ----------
    correct_answers: List[Dict[str, Any]] = []
    for q in questions:
        correct_answers.append(
            {
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "type": q.get("type", "mcq"),
                # Include options for MCQs so we can show them in results
                "options": q.get("options", []) if q.get("type", "mcq").lower() == "mcq" else None,
            }
        )

    # 1) Base evaluation: MCQs scored; free-form preserved (is_correct=False)
    base_result = evaluate_test(effective_answers, correct_answers)
    details: List[Dict[str, Any]] = list(base_result.get("details", []))

    # 2) Optional free-form & code grading (rules/LLM + runner), non-disruptive
    ff_points_sum = 0.0
    ff_max_sum = 0.0

    code_summary_total = 0
    code_summary_passed = 0
    total_points_sum = 0.0
    total_points_max = 0.0

    for i, det in enumerate(details):
        # Ensure type is visible to the UI
        try:
            qtype = (
                str(
                    (questions[i].get("type") if i < len(questions) else correct_answers[i].get("type", "mcq"))
                )
                .lower()
                .strip()
            )
        except Exception:
            qtype = "mcq"
        det["type"] = qtype

        # Carry ID through if provided by frontend/question bank
        if i < len(questions) and "id" in questions[i]:
            det["question_id"] = questions[i]["id"]

        # Preserve original fields
        question_text = _norm_text(correct_answers[i].get("question", "")) if i < len(correct_answers) else ""
        submitted_text = _norm_text(effective_answers[i].get("answer", "")) if i < len(effective_answers) else ""

        if qtype == "mcq":
            # ✅ For MCQs: Ensure correct answer is shown for wrong answers
            correct_ans = correct_answers[i].get("correct_answer", "")
            options = correct_answers[i].get("options")
            det["score"] = 1.0 if det.get("is_correct", False) else 0.0
            det["max_score"] = 1.0
            
            if not det.get("is_correct", False):
                # Show the correct answer in the explanation
                if correct_ans:
                    det["explanation"] = f"Incorrect. The correct answer is: {correct_ans}"
                    det["correct_answer"] = correct_ans  # Add correct answer field for UI
            
            # Include options for all MCQs (for UI display)
            if options and isinstance(options, list):
                det["options"] = options
            total_points_sum += float(det["score"])
            total_points_max += float(det["max_score"])
            continue

        # ---------------- Scenario / Free-form auto-grading (GPT-based for scenarios) ---------------- 
        if qtype in {"scenario", "freeform", "free-form", "text"}:
            # ✅ ALWAYS use GPT evaluator for scenario questions (no keyword matching)
            if qtype == "scenario":
                try:
                    from app.logic.scenario_evaluator import evaluate_scenario_with_leniency
                    gpt_result = evaluate_scenario_with_leniency(
                        question=question_text,
                        candidate_answer=submitted_text,
                        max_score=float(FREEFORM_MAX_POINTS),
                        ideal_answer=_norm_text((questions[i] if i < len(questions) else {}).get("ideal_answer", "")),
                        rubric=(questions[i] if i < len(questions) else {}).get("rubric"),
                    )
                    
                    auto_points = float(gpt_result.get("score", 0.0) or 0.0)
                    auto_max = float(FREEFORM_MAX_POINTS)
                    auto_feedback = str(gpt_result.get("explanation", "Evaluated by AI."))
                    
                    # Update detail with GPT evaluation results
                    det["auto_points"] = round(auto_points, 2)
                    det["auto_max"] = int(auto_max)
                    det["auto_feedback"] = auto_feedback
                    det["score"] = round(auto_points, 2)
                    det["max_score"] = int(auto_max)
                    det["is_correct"] = bool(gpt_result.get("is_correct", False))
                    det["explanation"] = auto_feedback
                    det["confidence"] = float(gpt_result.get("confidence", 0.0) or 0.0)
                    det["ai_reasoning"] = auto_feedback
                    
                    # Update aggregates
                    ff_points_sum += max(0.0, min(auto_points, auto_max))
                    ff_max_sum += auto_max
                    
                except Exception as e:
                    # Fallback if GPT evaluation fails
                    det["auto_points"] = 0.0
                    det["auto_max"] = int(FREEFORM_MAX_POINTS)
                    det["auto_feedback"] = f"Evaluation error: {str(e)[:100]}"
                    det["score"] = 0.0
                    det["max_score"] = int(FREEFORM_MAX_POINTS)
                    det["is_correct"] = False
                    det["explanation"] = det["auto_feedback"]
            else:
                # For other free-form types (not scenario), use existing logic
                auto_points = 0.0
                auto_max = float(FREEFORM_MAX_POINTS)
                auto_feedback_parts: List[str] = []

                if USE_RULES and _rules_grader:
                    try:
                        r = _rules_grader(
                            question=question_text,
                            answer=submitted_text,
                            max_points=FREEFORM_MAX_POINTS,
                        )
                        rp = float(r.get("points", 0.0) or 0.0)
                        rmax = float(r.get("max_points", FREEFORM_MAX_POINTS) or FREEFORM_MAX_POINTS)
                        rfb = _norm_text(r.get("feedback", ""))
                        auto_points = max(auto_points, rp)
                        auto_max = max(auto_max, rmax)
                        if rfb:
                            auto_feedback_parts.append(f"[rules] {rfb}")
                        if isinstance(r.get("rubric"), dict):
                            det["rubric_breakdown"] = r["rubric"]
                    except Exception:
                        pass

                if USE_LLM and _llm_grader:
                    try:
                        l = _llm_grader(
                            question=question_text,
                            answer=submitted_text,
                            max_points=FREEFORM_MAX_POINTS,
                        )
                        lp = float(l.get("points", 0.0) or 0.0)
                        lmax = float(l.get("max_points", FREEFORM_MAX_POINTS) or FREEFORM_MAX_POINTS)
                        lfb = _norm_text(l.get("feedback", ""))
                        if lp > auto_points:
                            auto_points = lp
                        auto_max = max(auto_max, lmax)
                        if lfb:
                            auto_feedback_parts.append(f"[llm] {lfb}")
                        if isinstance(l.get("rubric"), dict):
                            det["rubric_breakdown"] = l["rubric"]
                    except Exception:
                        pass

                if (USE_RULES and _rules_grader) or (USE_LLM and _llm_grader):
                    ff_points_sum += max(0.0, min(auto_points, auto_max))
                    ff_max_sum += auto_max

                det["auto_points"] = round(float(auto_points), 2)
                det["auto_max"] = int(auto_max)
                det["auto_feedback"] = " ".join(auto_feedback_parts).strip()
                det["score"] = det["auto_points"]
                det["max_score"] = det["auto_max"]
                if ff_max_sum > 0:
                    det["is_correct"] = det.get("is_correct", False) or (float(auto_points) >= (0.6 * auto_max))

        # ---------------- Code runner (optional) ----------------
        if qtype == "code" and CODE_RUNNER_ENABLED and _runner_run and submitted_text:
            # Figure out language preference: payload > question default > env default
            lang = _norm_text(
                effective_answers[i].get("language") or questions[i].get("language") or "python"
            )

            # Normalize tests (if present on the question)
            tcases = _code_tests_from_question(questions[i]) if i < len(questions) else []
            # Per-question limits (optional)
            q_time = (
                int(questions[i].get("time_limit_sec", RUNNER_TIME_LIMIT)) if i < len(questions) else RUNNER_TIME_LIMIT
            )
            q_mem = (
                int(questions[i].get("memory_limit_mb", RUNNER_MEM_MB)) if i < len(questions) else RUNNER_MEM_MB
            )

            payload = {
                "language": lang,
                "source": submitted_text,
                "tests": tcases,
                "time_limit_sec": q_time,
                "memory_limit_mb": q_mem,
            }

            try:
                runner_result = await run_in_threadpool(_runner_run, payload)
                # Attach raw tests for UI
                if isinstance(runner_result, dict) and isinstance(runner_result.get("tests"), list):
                    det["tests"] = runner_result["tests"]

                    total = len(runner_result["tests"])
                    passed = sum(1 for t in runner_result["tests"] if t.get("ok"))
                    code_summary_total += total
                    code_summary_passed += passed

                    # If there are tests, map to score/max_score (non-breaking)
                    if total > 0:
                        det["score"] = passed
                        det["max_score"] = total
                        # upgrade is_correct if all tests passed
                        det["is_correct"] = bool(passed == total)

                    # Provide a short explanation if some tests failed (first failure)
                    if total > 0 and passed < total:
                        first_fail = next((t for t in runner_result["tests"] if not t.get("ok")), None)
                        if first_fail:
                            mode = (first_fail.get("match_result") or {}).get("mode")
                            exp = (first_fail.get("match_result") or {}).get("expected")
                            det["explanation"] = (
                                det.get("explanation") or f"One or more tests failed. First failure mode: {mode or 'n/a'}."
                            )
                            # Append expected if available (short)
                            if isinstance(exp, str) and len(exp.strip()) <= 120:
                                det["explanation"] += f" Expected: {exp.strip()}"
                # If compile failed, reflect in explanation without throwing
                if isinstance(runner_result, dict) and isinstance(runner_result.get("compile"), dict):
                    if not runner_result["compile"].get("ok", True):
                        msg = runner_result["compile"].get("stderr") or runner_result["compile"].get("stdout") or "Compilation error"
                        det["explanation"] = (det.get("explanation") or "") + (
                            (" " if det.get("explanation") else "") + f"[compile] {msg[:240]}"
                        )
                        det["is_correct"] = False
            except Exception as e:
                # Don’t fail submission if runner has issues
                det["explanation"] = (det.get("explanation") or "") + (
                    (" " if det.get("explanation") else "") + f"[runner] {str(e)[:240]}"
                )
                det["is_correct"] = det.get("is_correct", False)

        if qtype == "code" and "score" not in det:
            det["score"] = 1.0 if det.get("is_correct", False) else 0.0
            det["max_score"] = 1.0

        try:
            det_score = float(det.get("score", 0.0) or 0.0)
            det_max = float(det.get("max_score", 0.0) or 0.0)
        except Exception:
            det_score = 0.0
            det_max = 0.0
        if det_max > 0:
            total_points_sum += max(0.0, min(det_score, det_max))
            total_points_max += det_max

    # 3) Score model:
    # - unified_total_percent includes ALL graded question types.
    # - mcq_percent is kept for backward compatibility/analytics.
    mcq_total = sum(1 for c in correct_answers if str(c.get("type", "mcq")).lower() == "mcq")
    raw_mcq = int(base_result.get("total_score", 0))
    mcq_percent = round(float((raw_mcq / mcq_total) * 100) if mcq_total > 0 else 0.0, 2)
    unified_total_percent = round((total_points_sum / total_points_max) * 100.0, 2) if total_points_max > 0 else 0.0

    # 4) Persist submission with COMPLETE test data
    submission_doc = {
        "_id": uuid.uuid4().hex,
        "testId": test["_id"],
        "candidateId": test["candidateId"],
        "answers": effective_answers,  # Candidate's submitted answers
        "questions": questions,  # ✅ Complete questions from the test
        "score": unified_total_percent,
        "total_score": unified_total_percent,
        "mcq_score": mcq_percent,
        "details": details,  # ✅ Complete evaluation details (question, answer, is_correct, explanation, etc.)
        "submittedAt": datetime.utcnow(),
        "needs_marking": False,  # SMART tests are auto-graded
        "type": "smart",
        # ✅ Include test metadata
        "testType": "smart",
        "questionCount": len(questions),
        "questionSeed": test.get("questionSeed"),
        "profileSignature": test.get("profileSignature"),
        "questionSetFingerprint": test.get("questionSetFingerprint") or _question_set_fingerprint(questions),
        "auto_submitted": effective_forced_by_expiry,
        "submit_reason": "expired_force_submit" if effective_forced_by_expiry else "manual_submit",
        "durationMinutes": test.get("testDurationMinutes") or invite.get("testDurationMinutes"),
        "ownerUserId": test.get("ownerUserId") or invite.get("ownerUserId"),
        "invitedByUserId": test.get("invitedByUserId") or invite.get("invitedByUserId"),
        "recruiterName": test.get("recruiterName") or invite.get("recruiterName"),
        "recruiterEmail": test.get("recruiterEmail") or invite.get("recruiterEmail"),
    }

    # Include free-form grading summary for reporting (non-breaking)
    if ff_max_sum > 0:
        submission_doc["freeform"] = {
            "points": round(ff_points_sum, 2),
            "max": int(ff_max_sum),
            "percent": round((ff_points_sum / ff_max_sum) * 100.0, 2),
        }

    # Optional code summary (non-breaking)
    if code_summary_total > 0:
        submission_doc["code_runner"] = {
            "passed": code_summary_passed,
            "total": code_summary_total,
            "percent": round((code_summary_passed / code_summary_total) * 100.0, 2),
        }
    submission_doc["total_marks"] = {
        "obtained": round(total_points_sum, 2),
        "max": round(total_points_max, 2),
        "percent": unified_total_percent,
    }

    try:
        await db.test_submissions.insert_one(submission_doc)
    except DuplicateKeyError:
        raise _one_test_conflict("Duplicate submission detected for this test.")

    # Mark test + invite
    await db.tests.update_one({"_id": test["_id"]}, {"$set": {"status": "submitted"}})
    await db.test_invites.update_one({"token": req.token}, {"$set": {"status": "completed"}})

    # Update candidate profile with latest test_score (%)
    try:
        await db.parsed_resumes.update_one({"_id": test["candidateId"]}, {"$set": {"test_score": unified_total_percent}})
    except Exception:
        pass  # not fatal

    return SubmitTestResponse(
        test_id=test["_id"],
        candidate_id=test["candidateId"],
        score=unified_total_percent,
        details=submission_doc["details"],
    )


@router.post("/autosave", response_model=AutosaveResponse)
async def autosave_test(req: AutosaveRequest):
    """
    Save in-progress answers server-side to prevent answer loss.
    If deadline already passed, immediately force-submit with server-saved answers.
    """
    test = await db.tests.find_one({"token": req.token})
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    if test.get("status") == "submitted":
        return AutosaveResponse(
            ok=True,
            status="submitted",
            saved_at=_to_utc_iso_z(datetime.utcnow()) or serialize_utc(datetime.utcnow()),
            auto_submitted=False,
        )

    questions: List[Dict[str, Any]] = test.get("questions", [])
    total_questions = len(questions)
    incoming_answers = _normalize_answers(req.answers, total_questions)
    existing_draft = _normalize_answers(test.get("draftAnswers", []), total_questions)
    merged = _merge_answers(incoming_answers, existing_draft, total_questions)

    now = datetime.utcnow()
    await db.tests.update_one(
        {"_id": test["_id"], "status": {"$ne": "submitted"}},
        {"$set": {"draftAnswers": merged, "draftSavedAt": now}},
    )

    expires_dt = _parse_utc_datetime(test.get("expiresAt"))
    if expires_dt and now > expires_dt:
        # Force submit using server-draft answers to lock deadline.
        await submit_test(SubmitTestRequest(token=req.token, answers=[]), forced_by_expiry=True)
        return AutosaveResponse(
            ok=True,
            status="submitted",
            saved_at=_to_utc_iso_z(now) or serialize_utc(now),
            auto_submitted=True,
        )

    return AutosaveResponse(
        ok=True,
        status="active",
        saved_at=_to_utc_iso_z(now) or serialize_utc(now),
        auto_submitted=False,
    )


async def sweep_expired_active_tests(limit: int = 100) -> Dict[str, int]:
    """
    Server-authoritative expiry sweep.
    Idempotent because submit path is protected by unique submission constraints.
    """
    now = datetime.utcnow()
    query = {"status": "active", "expiresAt": {"$lte": now}}
    docs = await db.tests.find(query).limit(max(1, int(limit))).to_list(length=max(1, int(limit)))
    forced = 0
    skipped = 0
    for test in docs:
        token = str(test.get("token") or "").strip()
        if not token:
            skipped += 1
            continue
        try:
            await submit_test(SubmitTestRequest(token=token, answers=[]), forced_by_expiry=True)
            forced += 1
        except Exception as e:
            logger.exception("expired test sweep force-submit failed token=%s err=%s", token, e)
            skipped += 1
    return {"checked": len(docs), "forced": forced, "skipped": skipped}


# =======================
# NEW: Manual grading API
# =======================


class QuestionGrade(BaseModel):
    question_index: int = Field(..., ge=0, description="Index of the question (0-based)")
    score: float = Field(..., ge=0, description="Score for this question (0-100)")
    feedback: Optional[str] = Field(default=None, description="Optional feedback for this answer")


class GradeRequest(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Final percent score to record for this attempt")
    question_grades: Optional[List[QuestionGrade]] = Field(
        default=None,
        description="Optional per-question scores for detailed marking"
    )


class GradeResponse(BaseModel):
    ok: bool
    attempt_id: str
    score: float
    candidate_id: str


@router.post("/grade/{attempt_id}", response_model=GradeResponse)
async def grade_attempt(
    attempt_id: str = Path(..., description="test_submissions._id"),
    body: GradeRequest = None,
    current=Depends(get_current_user),
):
    """
    Manually set the score for a custom test submission, and update the candidate's test_score.
    """
    attempt = await db.test_submissions.find_one({"_id": attempt_id})
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    # Only custom tests should be manually graded
    if attempt.get("type") != "custom":
        raise HTTPException(status_code=400, detail="Only custom test attempts can be manually graded")

    if body is None:
        raise HTTPException(status_code=400, detail="Request body is required")

    owner_id = str(getattr(current, "id", None) or "")
    candidate_id = attempt.get("candidateId")
    if candidate_id:
        owned_candidate = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
        if not owned_candidate:
            raise HTTPException(status_code=403, detail="Not authorized to grade this attempt")

    score = float(body.score)
    
    # ✅ Update details with per-question grades if provided
    details = list(attempt.get("details", []))
    computed_scores: List[float] = []
    if body.question_grades:
        for q_grade in body.question_grades:
            idx = q_grade.question_index
            if 0 <= idx < len(details):
                q_score = max(0.0, min(100.0, float(q_grade.score)))
                details[idx]["score"] = q_score
                details[idx]["max_score"] = 100.0  # Normalize to 100
                details[idx]["is_correct"] = q_score >= 60.0  # Consider >= 60% as correct
                if q_grade.feedback:
                    details[idx]["feedback"] = q_grade.feedback
                    details[idx]["explanation"] = q_grade.feedback
                computed_scores.append(q_score)
    fully_graded = bool(details) and len(computed_scores) == len(details)
    if computed_scores and fully_graded:
        score = round(sum(computed_scores) / len(computed_scores), 2)
    
    # ✅ Update submission with score and detailed grades
    update_data = {
        "details": details,  # ✅ Save updated details with per-question scores
        "manual_marking": {
            "question_count": len(computed_scores),
            "computed_from_questions": bool(computed_scores),
            "fully_graded": fully_graded,
        },
    }
    if fully_graded:
        update_data["score"] = score
        update_data["total_score"] = score
        update_data["needs_marking"] = False
        update_data["gradedAt"] = datetime.utcnow()
    else:
        # Keep the attempt pending until every question has a mark.
        update_data["needs_marking"] = True
    
    await db.test_submissions.update_one(
        {"_id": attempt_id},
        {"$set": update_data},
    )

    if candidate_id:
        try:
            # ✅ Update candidate profile with final score
            if fully_graded:
                await db.parsed_resumes.update_one({"_id": candidate_id}, {"$set": {"test_score": score}})
        except Exception:
            pass

    return GradeResponse(
        ok=True,
        attempt_id=attempt_id,
        score=score if fully_graded else float(attempt.get("score", 0.0) or 0.0),
        candidate_id=candidate_id or "",
    )


# =========================================
# NEW: Lightweight history/preview endpoints
# =========================================


def _attach_candidate_stub(doc: Dict[str, Any], cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cstub = None
    if cand:
        cstub = {
            "_id": cand.get("_id"),
            "name": _name_from_candidate_doc(cand),
            "email": _email_from_candidate_doc(cand),
            "resume": {"email": cand.get("resume", {}).get("email")} if isinstance(cand.get("resume"), dict) else {},
        }
    needs_marking = bool(doc.get("needs_marking", False))
    if needs_marking:
        status = "pending_evaluation"
    elif bool(doc.get("auto_submitted")):
        status = "auto_submitted"
    else:
        status = "completed"

    out = {
        "_id": doc.get("_id"),
        "candidateId": doc.get("candidateId"),
        "type": doc.get("type", "smart"),
        "testType": doc.get("testType", doc.get("type", "smart")),
        "score": doc.get("score"),
        "status": status,
        "submitted_at": _to_utc_iso_z(doc.get("submittedAt")) or doc.get("submittedAt"),
        "submittedAt": _to_utc_iso_z(doc.get("submittedAt")) or doc.get("submittedAt"),
        "created_at": _to_utc_iso_z(doc.get("submittedAt")) or doc.get("submittedAt"),  # for sorting in UI
        "candidate": cstub,
        "needs_marking": needs_marking,
        "questions": doc.get("questions", []),
        "answers": doc.get("answers", []),
        "details": doc.get("details", []),
    }
    return out


@router.get("/history/all")
async def get_history_all(current=Depends(get_current_user)):
    """
    Return recent attempts for preview, plus the subset that needs manual marking.
    Shape aligns with UI expectations:
      { attempts: [...], needs_marking: [...] }
    """
    owner_id = str(getattr(current, "id", None))
    # Scope to current owner's candidates to avoid data leakage.
    owned_ids = await db.parsed_resumes.distinct("_id", {"ownerUserId": owner_id})
    if not owned_ids:
        return {"attempts": [], "needs_marking": []}

    # Pull recent submissions for owned candidates only
    cur = db.test_submissions.find({"candidateId": {"$in": owned_ids}}).sort("submittedAt", -1).limit(100)
    subs = [s async for s in cur]

    # Fetch candidates in one pass
    cand_ids = list({s.get("candidateId") for s in subs if s.get("candidateId")})
    cand_map: Dict[str, Dict[str, Any]] = {}
    if cand_ids:
        async for c in db.parsed_resumes.find({"_id": {"$in": cand_ids}, "ownerUserId": owner_id}):
            cand_map[c["_id"]] = c

    attempts_raw = [_attach_candidate_stub(s, cand_map.get(s.get("candidateId"))) for s in subs]
    # Defensive de-duplication by attempt id.
    attempts_map: Dict[str, Dict[str, Any]] = {}
    for a in attempts_raw:
        aid = str(a.get("_id") or "")
        if aid and aid not in attempts_map:
            attempts_map[aid] = a
    attempts = list(attempts_map.values())
    needs = [a for a in attempts if any(x for x in subs if x["_id"] == a["_id"] and x.get("needs_marking"))]

    return {
        "attempts": attempts,
        "needs_marking": needs,
    }


@router.get("/history/{candidate_id}")
async def get_history_for_candidate(candidate_id: str, current=Depends(get_current_user)):
    """
    Return attempts for a specific candidate, newest first.
    """
    owner_id = str(getattr(current, "id", None))
    cand = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id})
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    cur = db.test_submissions.find({"candidateId": candidate_id}).sort("submittedAt", -1).limit(50)
    subs = [s async for s in cur]
    attempts = [_attach_candidate_stub(s, cand) for s in subs]
    return {"attempts": attempts}
