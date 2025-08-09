# app/models/auto_test_models.py

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# -------------------------
# Your original models (unchanged)
# -------------------------

class Candidate(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    email: Optional[str] = ""
    phone: Optional[str] = None
    skills: List[str]
    experience: Optional[str] = ""
    resume_text: Optional[str] = ""
    job_role: Optional[str] = ""  # ✅ Used for filtering & test generation

    class Config:
        allow_population_by_field_name = True  # lets us pass _id or id


class TestQuestion(BaseModel):
    type: str  # e.g., "mcq", "code", "scenario"
    question: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None  # for mcq -> string; for others -> None


class TestSubmission(BaseModel):
    candidate_id: str
    answers: List[str]
    correct_answers: List[str]
    score: Optional[float] = None
    submitted_at: Optional[str] = None


# -------------------------
# Additional models for the new tests flow (added, not replacing anything)
# -------------------------

# -- Invite --

class TestInviteCreate(BaseModel):
    candidate_id: str = Field(..., description="Candidate _id from parsed_resumes")
    subject: Optional[str] = "Your SmartHirex Assessment"
    # Optional custom HTML; the backend replaces {TEST_LINK} or {{TEST_LINK}} with the real link.
    body_html: Optional[str] = None


class TestInviteOut(BaseModel):
    invite_id: str
    token: str
    test_link: str
    email: str
    sent: bool
    expires_at: datetime


# -- Start Test --

class StartTestRequest(BaseModel):
    token: str


class StartTestResponse(BaseModel):
    test_id: str
    candidate_id: str
    # Keep it generic so the generator can evolve; still compatible with TestQuestion
    questions: List[Dict[str, Any]]


# -- Submit Test --

class SubmitTestRequest(BaseModel):
    token: str
    # Evaluator expects answers like [{"answer": "..."}] to support MCQ/free-form
    answers: List[Dict[str, Any]]


class SubmitTestResponse(BaseModel):
    test_id: str
    candidate_id: str
    score: float
    details: List[Dict[str, Any]]
