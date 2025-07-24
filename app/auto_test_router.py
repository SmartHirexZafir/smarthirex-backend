# ✅ File: app/auto_test_router.py

from fastapi import APIRouter, UploadFile, File
from typing import List
import os
import docx2txt

from app.models.auto_test_models import Candidate, TestSubmission
from app.logic.generator import generate_test
from app.logic.evaluator import evaluate_test
from app.logic.resume_parser import parse_resume_file  # ✅ FIXED import

router = APIRouter()

stored_candidates: List[Candidate] = []

@router.post("/upload-docx")
async def upload_docx_resumes(file: List[UploadFile] = File(...)):
    """
    Accepts multiple DOCX files, parses their content, and stores them as candidates.
    """
    global stored_candidates
    stored_candidates.clear()

    for upload in file:
        if upload.filename.endswith(".docx"):
            contents = await upload.read()
            parsed_data = parse_resume_file(upload.filename, contents)
            candidate = Candidate(
                id=upload.filename,
                name=parsed_data.get("name", "Unknown"),
                experience=f"{parsed_data.get('experience', 0)} years",
                skills=parsed_data.get("skills", []),
                job_role="Unknown"
            )
            stored_candidates.append(candidate)

    return {"message": f"{len(stored_candidates)} DOCX resumes processed and stored."}


@router.post("/generate-test")
def create_test(candidate: Candidate):
    """
    Generates a test based on candidate's experience, skills, and job role.
    """
    questions = generate_test(candidate)
    return {
        "candidate_id": candidate.id,
        "job_role": candidate.job_role,
        "questions": questions
    }

@router.post("/submit-test")
def submit_test(submission: TestSubmission):
    """
    Evaluates a submitted test against correct answers and returns score.
    """
    correct_answers = [answer.dict() for answer in submission.correct_answers]
    submitted_answers = [answer.dict() for answer in submission.submitted_answers]
    score = evaluate_test(submitted_answers, correct_answers)
    return {"score": score, "status": "submitted"}
