from fastapi import APIRouter, HTTPException, Path, Body
from typing import Literal
from app.utils.mongo import db

router = APIRouter()

@router.get("/{candidate_id}")
async def get_candidate(candidate_id: str = Path(...)):
    """
    Fetch full candidate detail by ID.
    Ensures all expected frontend fields are populated, even with defaults.
    """
    candidate = await db.parsed_resumes.find_one({"_id": candidate_id})
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Resume object & defaults
    candidate.setdefault("resume", {})
    resume = candidate["resume"]

    resume.setdefault("summary", "")
    resume.setdefault("education", [])
    resume.setdefault("workHistory", [])
    resume.setdefault("projects", [])
    resume.setdefault("filename", candidate.get("filename", "resume.pdf"))
    resume.setdefault("url", candidate.get("resume_url", ""))  # Optional: download link

    # Analysis fields
    candidate["score"] = candidate.get("score", 0)
    candidate["testScore"] = candidate.get("testScore", 0)
    candidate["matchedSkills"] = candidate.get("matchedSkills", [])
    candidate["missingSkills"] = candidate.get("missingSkills", [])
    candidate["strengths"] = candidate.get("strengths", [])
    candidate["redFlags"] = candidate.get("redFlags", [])
    candidate["selectionReason"] = candidate.get("selectionReason", "")
    candidate["rank"] = candidate.get("rank", 0)

    # Profile info
    candidate["name"] = candidate.get("name", "Unnamed Candidate")
    candidate["location"] = candidate.get("location", "N/A")
    candidate["currentRole"] = candidate.get("currentRole", "N/A")
    candidate["company"] = candidate.get("company", "N/A")
    candidate["avatar"] = candidate.get("avatar", "")

    # Model output fields
    candidate["category"] = candidate.get("category") or candidate.get("predicted_role", "")
    candidate["confidence"] = candidate.get("confidence", 0)
    candidate["match_reason"] = candidate.get("match_reason", "ML classified")
    candidate["semantic_score"] = candidate.get("semantic_score", None)

    return candidate


@router.patch("/{candidate_id}/status")
async def update_candidate_status(
    candidate_id: str = Path(...),
    status: Literal["shortlisted", "rejected", "new", "interviewed"] = Body(..., embed=True)
):
    """
    Update candidate status (shortlisted, rejected, etc).
    """
    result = await db.parsed_resumes.update_one(
        {"_id": candidate_id},
        {"$set": {"status": status}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Candidate not found or status unchanged")

    return {"message": f"Candidate status updated to '{status}'."}
