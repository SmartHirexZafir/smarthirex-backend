# app/models/auto_test_models.py

from pydantic import BaseModel, Field
from typing import List, Optional


class Candidate(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    email: Optional[str] = ""
    phone: Optional[str] = None
    skills: List[str]
    experience: Optional[str] = ""
    resume_text: Optional[str] = ""
    job_role: Optional[str] = ""  # ✅ Used for filtering & test generation


class TestQuestion(BaseModel):
    type: str  # e.g., "mcq", "code", "scenario"
    question: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None


class TestSubmission(BaseModel):
    candidate_id: str
    answers: List[str]
    correct_answers: List[str]
    score: Optional[float] = None
    submitted_at: Optional[str] = None
