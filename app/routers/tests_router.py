# app/routers/tests_router.py
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Literal

from fastapi import APIRouter, HTTPException, Path, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.utils.mongo import db  # AsyncIOMotorDatabase
from app.utils.emailer import send_email, render_invite_html
from app.models.auto_test_models import Candidate
from app.logic.generator import generate_test
from app.logic.evaluator import evaluate_test  # returns {"total_score": int, "details": [...]}
from app.routers.auth_router import get_current_user  # ✅ auth dependency

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

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
TEST_TOKEN_EXPIRY_MINUTES = int(os.getenv("TEST_TOKEN_EXPIRY_MINUTES", "60"))

# ---------- helpers ----------


def _extract_int_years(value: Optional[str]) -> int:
    if not value:
        return 0
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else 0


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
    exp = (
        doc.get("experience")
        or doc.get("total_experience")
        or doc.get("resume", {}).get("experience")
        or doc.get("resume", {}).get("totalExperience")
        or ""
    )
    years = _extract_int_years(str(exp))
    return f"{years} years"


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
    expires_at: datetime


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
    score: float  # percent (MCQs only – unchanged; 0.0 for ungraded custom)
    details: List[Dict[str, Any]]


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

    # ✅ ENFORCE: Check if candidate already has a test assigned (pending/active)
    existing_invite = await db.test_invites.find_one({
        "candidateId": req.candidate_id,
        "status": {"$in": ["pending", "active"]}
    })
    if existing_invite:
        existing_type = existing_invite.get("type", "smart")
        raise HTTPException(
            status_code=400,
            detail=f"Candidate already has a {existing_type} test assigned. Only one test type (Smart AI or Custom) is allowed per candidate."
        )

    # ✅ Also check if candidate has a completed test
    existing_test = await db.tests.find_one({
        "candidateId": req.candidate_id,
        "status": {"$in": ["active", "submitted"]}
    })
    if existing_test:
        existing_test_type = existing_test.get("type", "smart")
        raise HTTPException(
            status_code=400,
            detail=f"Candidate already has a {existing_test_type} test. Only one test type (Smart AI or Custom) is allowed per candidate."
        )

    name = _name_from_candidate_doc(candidate)
    role = _role_from_candidate_doc(candidate)

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

    # ✅ Parse scheduled_date_time from request
    scheduled_datetime: Optional[datetime] = None
    if req.scheduled_date_time:
        try:
            scheduled_datetime = datetime.fromisoformat(req.scheduled_date_time.replace('Z', '+00:00'))
            if scheduled_datetime.tzinfo is None:
                scheduled_datetime = scheduled_datetime.replace(tzinfo=datetime.now().astimezone().tzinfo)
            scheduled_datetime = scheduled_datetime.astimezone(datetime.now().astimezone().tzinfo).replace(tzinfo=None)
        except (ValueError, AttributeError):
            scheduled_datetime = None
    
    # ✅ Get test duration from request
    test_duration_minutes = req.test_duration_minutes or 60

    invite_doc = {
        "_id": uuid.uuid4().hex,
        "candidateId": candidate["_id"],
        "email": email,
        "name": name,
        "role": role,
        "experienceYears": _extract_int_years(_experience_string_from_doc(candidate)),
        "status": "pending",
        "token": token,
        "expiresAt": expires_at,
        "createdAt": datetime.utcnow(),
        "questionCount": question_count,
        "type": test_type,  # <-- NEW
        "custom": custom_payload,  # <-- NEW (optional)
        "composition": composition_payload,  # <-- NEW (optional, for Smart AI Test)
        # ✅ NEW: Scheduled timing and duration
        "scheduledDateTime": scheduled_datetime.isoformat() if scheduled_datetime else None,
        "testDurationMinutes": test_duration_minutes,
    }
    await db.test_invites.insert_one(invite_doc)

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
        expires_at=expires_at,
    )


@router.post("/start", response_model=StartTestResponse)
async def start_test(req: StartTestRequest):
    """
    Validate token & expiry, then generate a test based on sender's composition choices.

    ENHANCEMENTS:
    - If invite.type == "custom", return the custom questions instead of generating.
    - If invite.type == "smart", use composition parameters (mcq_count, scenario_count, code_count)
      provided by the sender. No experience-based logic - sender decides everything.
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
        try:
            from datetime import timezone
            # Parse the scheduled datetime string - handle 'Z' suffix
            scheduled_dt_str_clean = scheduled_dt_str.strip()
            if scheduled_dt_str_clean.endswith('Z'):
                scheduled_dt_str_clean = scheduled_dt_str_clean[:-1] + '+00:00'
            
            scheduled_dt = datetime.fromisoformat(scheduled_dt_str_clean)
            
            # Convert to UTC-naive for comparison with datetime.utcnow()
            if scheduled_dt.tzinfo is not None:
                # If timezone-aware, convert to UTC then remove timezone info
                scheduled_dt_utc = scheduled_dt.astimezone(timezone.utc).replace(tzinfo=None)
            else:
                # If already naive, assume it's UTC (as stored)
                scheduled_dt_utc = scheduled_dt
            
            # Compare UTC-naive datetimes - only block if current time is BEFORE scheduled time
            if now < scheduled_dt_utc:
                # Test not yet available - return scheduled time for countdown
                scheduled_iso = scheduled_dt_utc.isoformat()
                raise HTTPException(
                    status_code=403,
                    detail=f"Test is scheduled for {scheduled_iso}. Please wait until the scheduled time. SCHEDULED_DATETIME:{scheduled_iso}"
                )
            # If we reach here, now >= scheduled_dt_utc, so scheduled time has passed - allow test to proceed
        except (ValueError, AttributeError, TypeError):
            # Invalid date format, skip validation and allow test to proceed
            pass
    
    # ✅ Check if test has already started - if so, check 30-minute window
    existing_test = await db.tests.find_one({"token": req.token})
    if existing_test:
        started_at = existing_test.get("startedAt")
        if started_at:
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                if started_at.tzinfo:
                    started_at = started_at.replace(tzinfo=None)
            # Allow if started within last 30 minutes
            if (now - started_at).total_seconds() > 1800:  # 30 minutes
                raise HTTPException(
                    status_code=410,
                    detail="Test link expired. The test must be started within 30 minutes of availability."
                )
            # Test already started - return existing test
            expires_at = existing_test.get("expiresAt")
            if isinstance(expires_at, str):
                expires_at_str = expires_at
            elif expires_at:
                expires_at_str = expires_at.isoformat()
            else:
                expires_at_str = None
            return StartTestResponse(
                test_id=existing_test["_id"],
                candidate_id=existing_test["candidateId"],
                questions=existing_test.get("questions", []),
                scheduled_datetime=invite.get("scheduledDateTime"),
                expires_at=expires_at_str,
                duration_minutes=existing_test.get("testDurationMinutes"),
            )
    
    # ✅ Check general expiration
    if now > invite["expiresAt"]:
        raise HTTPException(status_code=410, detail="Invite link expired")

    candidate = await db.parsed_resumes.find_one({"_id": invite["candidateId"]})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

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
        candidate_model = Candidate(
            _id=candidate["_id"],
            name=_name_from_candidate_doc(candidate),
            email=_email_from_candidate_doc(candidate) or "",
            phone=None,
            skills=_skills_from_candidate_doc(candidate),
            experience=_experience_string_from_doc(candidate),
            resume_text=candidate.get("raw_text", ""),
            job_role=_role_from_candidate_doc(candidate),
        )
        
        # ✅ Fetch previous questions for this candidate to ensure uniqueness
        previous_questions: List[str] = []
        previous_tests = await db.tests.find({"candidateId": candidate["_id"]}).to_list(length=100)
        for prev_test in previous_tests:
            prev_questions = prev_test.get("questions", [])
            if isinstance(prev_questions, list):
                for q in prev_questions:
                    q_text = q.get("question", "")
                    if q_text and q_text not in previous_questions:
                        previous_questions.append(q_text)
        
        # Use composition parameters if provided, otherwise fallback to question_count
        composition = invite.get("composition")
        if composition:
            questions = generate_test(
                candidate_model,
                mcq_count=composition.get("mcq_count", 0),
                scenario_count=composition.get("scenario_count", 0),
                code_count=composition.get("code_count", 0),
                previous_questions=previous_questions if previous_questions else None,
            )
            # Calculate question_count from composition
            question_count = composition.get("mcq_count", 0) + composition.get("scenario_count", 0) + composition.get("code_count", 0)
            # Fallback to actual questions length if calculation fails
            if question_count == 0:
                question_count = len(questions) if questions else 4
        else:
            # Fallback for backward compatibility
            question_count = _sanitize_question_count(invite.get("questionCount", 4))
            questions = generate_test(
                candidate_model,
                question_count=question_count,
                previous_questions=previous_questions if previous_questions else None,
            )
    else:
        # custom path already set questions; still capture a sensible count
        question_count = len(questions) if questions else _sanitize_question_count(invite.get("questionCount", 4))

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
        "testDurationMinutes": test_duration_minutes,  # ✅ Store duration
    }
    await db.tests.insert_one(test_doc)

    if invite.get("status") == "pending":
        await db.test_invites.update_one({"_id": invite["_id"]}, {"$set": {"status": "active"}})

    return StartTestResponse(
        test_id=test_doc["_id"],
        candidate_id=candidate["_id"],
        questions=questions,
        scheduled_datetime=invite.get("scheduledDateTime"),
        expires_at=expires_at.isoformat(),
        duration_minutes=test_duration_minutes,
    )


@router.post("/submit", response_model=SubmitTestResponse)
async def submit_test(req: SubmitTestRequest):
    """
    Persist answers, grade with evaluator, return score + details.
    Also updates the candidate's 'test_score' (%) in parsed_resumes.

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

    # ---------- CUSTOM TEST PATH (manual marking) ----------
    if test_type == "custom":
        # Build lightweight details pairing question/answer for UI preview
        questions: List[Dict[str, Any]] = test.get("questions", [])
        details: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            details.append(
                {
                    "question": q.get("question", ""),
                    "type": q.get("type", "text"),
                    "answer": (req.answers[i].get("answer") if i < len(req.answers) else None),
                    "is_correct": None,
                }
            )

        # ✅ Save COMPLETE test data for custom tests
        submission_doc = {
            "_id": uuid.uuid4().hex,
            "testId": test["_id"],
            "candidateId": test["candidateId"],
            "answers": req.answers,  # Candidate's submitted answers
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
        }
        await db.test_submissions.insert_one(submission_doc)
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
    questions: List[Dict[str, Any]] = test.get("questions", [])
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
    base_result = evaluate_test(req.answers, correct_answers)
    details: List[Dict[str, Any]] = list(base_result.get("details", []))

    # 2) Optional free-form & code grading (rules/LLM + runner), non-disruptive
    ff_points_sum = 0.0
    ff_max_sum = 0.0

    code_summary_total = 0
    code_summary_passed = 0

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
        submitted_text = _norm_text(req.answers[i].get("answer", "")) if i < len(req.answers) else ""

        if qtype == "mcq":
            # ✅ For MCQs: Ensure correct answer is shown for wrong answers
            correct_ans = correct_answers[i].get("correct_answer", "")
            options = correct_answers[i].get("options")
            
            if not det.get("is_correct", False):
                # Show the correct answer in the explanation
                if correct_ans:
                    det["explanation"] = f"Incorrect. The correct answer is: {correct_ans}"
                    det["correct_answer"] = correct_ans  # Add correct answer field for UI
            
            # Include options for all MCQs (for UI display)
            if options and isinstance(options, list):
                det["options"] = options
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
                    )
                    
                    auto_points = float(gpt_result.get("score", 0.0) or 0.0)
                    auto_max = float(FREEFORM_MAX_POINTS)
                    auto_feedback = str(gpt_result.get("explanation", "Evaluated by AI."))
                    gpt_score_normalized = float(gpt_result.get("normalized_score", 0.0) or 0.0)
                    
                    # Update detail with GPT evaluation results
                    det["auto_points"] = round(auto_points, 2)
                    det["auto_max"] = int(auto_max)
                    det["auto_feedback"] = auto_feedback
                    det["score"] = round(auto_points, 2)
                    det["max_score"] = int(auto_max)
                    det["is_correct"] = bool(gpt_result.get("is_correct", False))
                    det["explanation"] = auto_feedback
                    det["confidence"] = float(gpt_result.get("confidence", 0.0) or 0.0)
                    
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
                req.answers[i].get("language") or questions[i].get("language") or "python"
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

    # 3) MCQ percent (unchanged)
    mcq_total = sum(1 for c in correct_answers if str(c.get("type", "mcq")).lower() == "mcq")
    raw_mcq = int(base_result.get("total_score", 0))
    percent = float((raw_mcq / mcq_total) * 100) if mcq_total > 0 else 0.0
    percent = round(percent, 2)

    # 4) Persist submission with COMPLETE test data
    submission_doc = {
        "_id": uuid.uuid4().hex,
        "testId": test["_id"],
        "candidateId": test["candidateId"],
        "answers": req.answers,  # Candidate's submitted answers
        "questions": questions,  # ✅ Complete questions from the test
        "score": percent,  # store MCQ percent (unchanged for UI compatibility)
        "details": details,  # ✅ Complete evaluation details (question, answer, is_correct, explanation, etc.)
        "submittedAt": datetime.utcnow(),
        "needs_marking": False,  # SMART tests are auto-graded
        "type": "smart",
        # ✅ Include test metadata
        "testType": "smart",
        "questionCount": len(questions),
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

    await db.test_submissions.insert_one(submission_doc)

    # Mark test + invite
    await db.tests.update_one({"_id": test["_id"]}, {"$set": {"status": "submitted"}})
    await db.test_invites.update_one({"token": req.token}, {"$set": {"status": "completed"}})

    # Update candidate profile with latest test_score (%)
    try:
        await db.parsed_resumes.update_one({"_id": test["candidateId"]}, {"$set": {"test_score": percent}})
    except Exception:
        pass  # not fatal

    return SubmitTestResponse(
        test_id=test["_id"],
        candidate_id=test["candidateId"],
        score=percent,
        details=submission_doc["details"],
    )


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

    score = float(body.score)
    
    # ✅ Update details with per-question grades if provided
    details = list(attempt.get("details", []))
    if body.question_grades:
        for q_grade in body.question_grades:
            idx = q_grade.question_index
            if 0 <= idx < len(details):
                details[idx]["score"] = float(q_grade.score)
                details[idx]["max_score"] = 100.0  # Normalize to 100
                details[idx]["is_correct"] = q_grade.score >= 60.0  # Consider >= 60% as correct
                if q_grade.feedback:
                    details[idx]["feedback"] = q_grade.feedback
                    details[idx]["explanation"] = q_grade.feedback
    
    # ✅ Update submission with score and detailed grades
    update_data = {
        "score": score,
        "needs_marking": False,
        "gradedAt": datetime.utcnow(),
        "details": details,  # ✅ Save updated details with per-question scores
    }
    
    await db.test_submissions.update_one(
        {"_id": attempt_id},
        {"$set": update_data},
    )

    candidate_id = attempt.get("candidateId")
    if candidate_id:
        try:
            # ✅ Update candidate profile with final score
            await db.parsed_resumes.update_one({"_id": candidate_id}, {"$set": {"test_score": score}})
        except Exception:
            pass

    return GradeResponse(ok=True, attempt_id=attempt_id, score=score, candidate_id=candidate_id or "")


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
    out = {
        "_id": doc.get("_id"),
        "candidateId": doc.get("candidateId"),
        "type": doc.get("type", "smart"),
        "score": doc.get("score"),
        "submitted_at": doc.get("submittedAt"),
        "created_at": doc.get("submittedAt"),  # for sorting in UI
        "candidate": cstub,
    }
    return out


@router.get("/history/all")
async def get_history_all(current=Depends(get_current_user)):
    """
    Return recent attempts for preview, plus the subset that needs manual marking.
    Shape aligns with UI expectations:
      { attempts: [...], needs_marking: [...] }
    """
    # Pull recent submissions
    cur = db.test_submissions.find({}).sort("submittedAt", -1).limit(100)
    subs = [s async for s in cur]

    # Fetch candidates in one pass
    cand_ids = list({s.get("candidateId") for s in subs if s.get("candidateId")})
    cand_map: Dict[str, Dict[str, Any]] = {}
    if cand_ids:
        async for c in db.parsed_resumes.find({"_id": {"$in": cand_ids}}):
            cand_map[c["_id"]] = c

    attempts = [_attach_candidate_stub(s, cand_map.get(s.get("candidateId"))) for s in subs]
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
    cur = db.test_submissions.find({"candidateId": candidate_id}).sort("submittedAt", -1).limit(50)
    subs = [s async for s in cur]
    cand = await db.parsed_resumes.find_one({"_id": candidate_id})
    attempts = [_attach_candidate_stub(s, cand) for s in subs]
    return {"attempts": attempts}
