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

    # Ensure resume structure
    candidate.setdefault("resume", {})
    resume = candidate["resume"]

    resume.setdefault("summary", "")
    resume.setdefault("education", [])
    resume.setdefault("workHistory", [])
    resume.setdefault("projects", [])
    resume.setdefault("filename", candidate.get("filename", "resume.pdf"))
    resume.setdefault("url", candidate.get("resume_url", ""))  # Optional: download link

    # Defaults for analysis
    candidate.setdefault("score", 0)
    candidate.setdefault("testScore", 0)
    candidate.setdefault("matchedSkills", [])
    candidate.setdefault("missingSkills", [])
    candidate.setdefault("strengths", [])
    candidate.setdefault("redFlags", [])
    candidate.setdefault("selectionReason", "")
    candidate.setdefault("rank", 0)

    # Contact & profile info
    candidate.setdefault("name", "Unnamed Candidate")  # ✅ fallback
    candidate.setdefault("location", "N/A")
    candidate.setdefault("currentRole", "N/A")
    candidate.setdefault("company", "N/A")
    candidate.setdefault("avatar", "")

    # ✅ Ensure model output fields for UI
    candidate.setdefault("category", candidate.get("predicted_role", ""))
    candidate.setdefault("confidence", 0)
    candidate.setdefault("match_reason", "ML classified")
    candidate.setdefault("semantic_score", None)

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
