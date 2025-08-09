# app/routers/tests_router.py
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.utils.mongo import db  # AsyncIOMotorDatabase
from app.utils.emailer import send_email, render_invite_html
from app.models.auto_test_models import Candidate
from app.logic.generator import generate_test
from app.logic.evaluator import evaluate_test  # returns {"total_score": int, "details": [...]}

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


# ---------- request/response models ----------

class InviteRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate _id from parsed_resumes")
    subject: Optional[str] = "Your SmartHirex Assessment"
    body_html: Optional[str] = None  # optional custom HTML; {TEST_LINK} will be replaced


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
    answers: List[Dict[str, Any]]


class SubmitTestResponse(BaseModel):
    test_id: str
    candidate_id: str
    score: float  # percent
    details: List[Dict[str, Any]]


# ---------- routes ----------

@router.post("/invite", response_model=InviteResponse)
async def create_invite(req: InviteRequest):
    """
    Create a test invite for a candidate and send an email with a secure link.
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

    questions = generate_test(candidate_model)

    test_doc = {
        "_id": uuid.uuid4().hex,
        "inviteId": invite["_id"],
        "candidateId": candidate["_id"],
        "token": req.token,
        "questions": questions,
        "status": "active",
        "startedAt": datetime.utcnow(),
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
    """
    test = await db.tests.find_one({"token": req.token})
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    # If already submitted, return the stored result (idempotent)
    if test.get("status") == "submitted":
        existing = await db.test_submissions.find_one({"testId": test["_id"]})
        if existing:
            # ensure candidate has the saved percentage
            try:
                await db.parsed_resumes.update_one(
                    {"_id": test["candidateId"]},
                    {"$set": {"test_score": float(existing["score"])}}
                )
            except Exception:
                pass
            return SubmitTestResponse(
                test_id=test["_id"],
                candidate_id=test["candidateId"],
                score=float(existing["score"]),
                details=existing["details"],
            )

    questions: List[Dict[str, Any]] = test.get("questions", [])
    correct_answers: List[Dict[str, Any]] = []
    for q in questions:
        correct_answers.append({
            "question": q.get("question", ""),
            "correct_answer": q.get("correct_answer", ""),
            "type": q.get("type", "mcq"),
        })

    # Your evaluator expects: answers = [{"answer": "..."}]
    result = evaluate_test(req.answers, correct_answers)  # {"total_score": int, "details": [...]}

    # Convert raw score to percentage based on number of MCQs
    mcq_total = sum(1 for c in correct_answers if str(c.get("type", "mcq")).lower() == "mcq")
    raw = int(result.get("total_score", 0))
    percent = float((raw / mcq_total) * 100) if mcq_total > 0 else 0.0
    # round to 2 decimals to look nice in UI
    percent = round(percent, 2)

    submission_doc = {
        "_id": uuid.uuid4().hex,
        "testId": test["_id"],
        "candidateId": test["candidateId"],
        "answers": req.answers,
        "score": percent,  # store percent
        "details": result.get("details", []),
        "submittedAt": datetime.utcnow(),
    }
    await db.test_submissions.insert_one(submission_doc)

    # Mark test + invite
    await db.tests.update_one({"_id": test["_id"]}, {"$set": {"status": "submitted"}})
    await db.test_invites.update_one({"token": req.token}, {"$set": {"status": "completed"}})

    # Update candidate profile with latest test_score (%)
    try:
        await db.parsed_resumes.update_one(
            {"_id": test["candidateId"]},
            {"$set": {"test_score": percent}}
        )
    except Exception:
        # not fatal for the response
        pass

    return SubmitTestResponse(
        test_id=test["_id"],
        candidate_id=test["candidateId"],
        score=percent,
        details=submission_doc["details"],
    )
