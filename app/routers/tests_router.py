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


class InviteRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate _id from parsed_resumes")
    subject: Optional[str] = "Your SmartHirex Assessment"
    body_html: Optional[str] = None  # optional custom HTML; {TEST_LINK} will be replaced
    # allow sender to choose number of questions
    question_count: Optional[int] = Field(default=4, ge=1, le=50)
    # NEW: test type & optional custom
    test_type: Optional[TestType] = Field(default="smart")
    custom: Optional[CustomTest] = None


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
    """
    candidate = await db.parsed_resumes.find_one({"_id": req.candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    email = _email_from_candidate_doc(candidate)
    if not email:
        raise HTTPException(status_code=400, detail="Candidate email not found")

    name = _name_from_candidate_doc(candidate)
    role = _role_from_candidate_doc(candidate)

    token = uuid.uuid4().hex
    expires_at = datetime.utcnow() + timedelta(minutes=TEST_TOKEN_EXPIRY_MINUTES)
    test_link = f"{FRONTEND_BASE_URL.rstrip('/')}/test/{token}"

    question_count = _sanitize_question_count(req.question_count)
    test_type: TestType = req.test_type or "smart"

    # persist custom payload when applicable
    custom_payload: Optional[Dict[str, Any]] = None
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
    }
    await db.test_invites.insert_one(invite_doc)

    html = req.body_html or render_invite_html(candidate_name=name, role=role, test_link=test_link)
    html = html.replace("{TEST_LINK}", test_link).replace("{{TEST_LINK}}", test_link)

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
    Validate token & expiry, then generate a tailored test by role/experience.

    ENHANCEMENTS:
    - If invite.type == "custom", return the custom questions instead of generating.
    """
    invite = await db.test_invites.find_one({"token": req.token})
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.get("status") not in {"pending", "active"}:
        raise HTTPException(status_code=400, detail=f"Invite status is {invite.get('status')}")
    if datetime.utcnow() > invite["expiresAt"]:
        raise HTTPException(status_code=410, detail="Invite link expired")

    candidate = await db.parsed_resumes.find_one({"_id": invite["candidateId"]})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Compute questions based on type
    test_type: TestType = invite.get("type", "smart")

    if test_type == "custom":
        custom = invite.get("custom") or {}
        questions = custom.get("questions") or []
        if not isinstance(questions, list) or not questions:
            # Fallback gracefully to generator if custom is empty/malformed
            test_type = "smart"

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
        # honor the sender-selected question count (default 4)
        question_count = _sanitize_question_count(invite.get("questionCount", 4))
        questions = generate_test(candidate_model, question_count=question_count)
    else:
        # custom path already set questions; still capture a sensible count
        question_count = len(questions)

    test_doc = {
        "_id": uuid.uuid4().hex,
        "inviteId": invite["_id"],
        "candidateId": candidate["_id"],
        "token": req.token,
        "questions": questions,
        "status": "active",
        "startedAt": datetime.utcnow(),
        "questionCount": question_count,
        "type": test_type,  # <-- NEW
    }
    await db.tests.insert_one(test_doc)

    if invite.get("status") == "pending":
        await db.test_invites.update_one({"_id": invite["_id"]}, {"$set": {"status": "active"}})

    return StartTestResponse(
        test_id=test_doc["_id"],
        candidate_id=candidate["_id"],
        questions=questions,
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

        submission_doc = {
            "_id": uuid.uuid4().hex,
            "testId": test["_id"],
            "candidateId": test["candidateId"],
            "answers": req.answers,
            "score": 0.0,  # not graded yet
            "details": details,
            "submittedAt": datetime.utcnow(),
            "needs_marking": True,
            "type": "custom",
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
            # keep as-is
            continue

        # ---------------- Scenario / Free-form auto-grading ----------------
        if qtype in {"scenario", "freeform", "free-form", "text"}:
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
                    # pass rubric if available
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

            # update aggregates if any free-form grading was active
            if (USE_RULES and _rules_grader) or (USE_LLM and _llm_grader):
                ff_points_sum += max(0.0, min(auto_points, auto_max))
                ff_max_sum += auto_max

            # attach non-breaking extras to detail (+ UI-friendly aliases)
            det["auto_points"] = round(float(auto_points), 2)
            det["auto_max"] = int(auto_max)
            det["auto_feedback"] = " ".join(auto_feedback_parts).strip()
            det["score"] = det["auto_points"]
            det["max_score"] = det["auto_max"]
            # Set pass/fail only if graders were enabled; else leave as-is
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

    # 4) Persist submission
    submission_doc = {
        "_id": uuid.uuid4().hex,
        "testId": test["_id"],
        "candidateId": test["candidateId"],
        "answers": req.answers,
        "score": percent,  # store MCQ percent (unchanged for UI compatibility)
        "details": details,
        "submittedAt": datetime.utcnow(),
        "needs_marking": False,  # SMART tests are auto-graded
        "type": "smart",
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


class GradeRequest(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Final percent score to record for this attempt")


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
    await db.test_submissions.update_one(
        {"_id": attempt_id},
        {
            "$set": {
                "score": score,
                "needs_marking": False,
                "gradedAt": datetime.utcnow(),
            }
        },
    )

    candidate_id = attempt.get("candidateId")
    if candidate_id:
        try:
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
