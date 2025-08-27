from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash
from app.logic.ml_interface import clean_text, model, vectorizer, label_encoder
from app.routers.auth_router import get_current_user  # ✅ added import
import uuid
import os

router = APIRouter()

ALLOWED_EXTS = {".pdf", ".doc", ".docx"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB (frontend copy says 10MB)

@router.post("/upload-resumes")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    current_user = Depends(get_current_user),  # ✅ inject current logged-in user
):
    parsed_resumes = []

    # counters to help UI show "X of Y"
    received_count = len(files)
    inserted_count = 0
    skipped_duplicates = 0
    skipped_unsupported = 0
    skipped_empty = 0
    skipped_too_large = 0
    skipped_parse_error = 0

    for file in files:
        # ✅ allow pdf, doc, docx (align with frontend)
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTS:
            skipped_unsupported += 1
            continue

        contents = await file.read()
        if not contents:
            skipped_empty += 1
            continue

        # ✅ enforce 10MB to match frontend hint/constraint
        if len(contents) > MAX_SIZE_BYTES:
            skipped_too_large += 1
            continue

        # Parse safely
        try:
            resume_data = parse_resume_file(file.filename, contents)
        except Exception:
            skipped_parse_error += 1
            continue

        raw_text = resume_data.get("raw_text", "") or ""
        if not raw_text.strip():
            skipped_empty += 1
            continue

        # 🚫 Skip duplicates (scoped per-user)
        if await check_duplicate_cv_hash(raw_text, owner_user_id=current_user.id):
            skipped_duplicates += 1
            continue

        # --- ML prediction (unchanged) ---
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max()
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # ✅ Assign required metadata
        resume_data["_id"] = str(uuid.uuid4())
        resume_data["content_hash"] = compute_cv_hash(raw_text)
        resume_data["predicted_role"] = predicted_label
        resume_data["category"] = predicted_label
        resume_data["confidence"] = round(confidence * 100, 2)
        resume_data["filename"] = file.filename
        resume_data["experience"] = resume_data.get("experience", 0)

        # 🔁 IMPORTANT: aap ki requirement ke mutabiq missing name => "No Name"
        resume_data["name"] = (resume_data.get("name") or "").strip() or "No Name"

        resume_data["location"] = resume_data.get("location") or "N/A"
        resume_data["raw_text"] = raw_text

        # ✅ Ensure required profile fields
        resume_data["email"] = resume_data.get("email") or None
        resume_data["phone"] = resume_data.get("phone") or None
        resume_data["skills"] = resume_data.get("skills") or []
        resume_data["resume_url"] = resume_data.get("resume_url", "")  # optional
        resume_data["currentRole"] = resume_data.get("currentRole") or "N/A"
        resume_data["company"] = resume_data.get("company") or "N/A"

        # ✅ Ensure resume subfields (summary, education, workHistory, projects)
        resume_data["resume"] = resume_data.get("resume", {})
        resume = resume_data["resume"]
        resume["summary"] = resume.get("summary", "")
        resume["education"] = resume.get("education", [])
        resume["workHistory"] = resume.get("workHistory", [])
        resume["projects"] = resume.get("projects", [])
        resume["filename"] = file.filename
        resume["url"] = resume_data.get("resume_url", "")

        # 🧹 Remove outdated fields if exist
        for key in ["matchedSkills", "missingSkills", "score", "testScore", "rank", "filter_skills"]:
            resume_data.pop(key, None)

        # ✅ Assign ownership to current user
        resume_data["ownerUserId"] = str(current_user.id)

        # ✅ Save to database
        await db.parsed_resumes.insert_one(resume_data)
        inserted_count += 1

        # ✅ Minimal safe fields for frontend
        parsed_resumes.append({
            "_id": resume_data["_id"],
            "name": resume_data["name"],  # "No Name" if missing
            "predicted_role": resume_data["predicted_role"],
            "category": resume_data["category"],  # ➕ included for frontend convenience
            "experience": resume_data["experience"],
            "location": resume_data["location"],
            "confidence": resume_data["confidence"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "skills": resume_data["skills"],
            "resume_url": resume_data["resume_url"],
            "filename": resume_data["filename"]
        })

    return {
        "message": "Resumes processed successfully",
        "resumes": parsed_resumes,          # keep existing shape for frontend
        # ➕ extra counters (non-breaking) — helpful for progress UI
        "received": received_count,
        "inserted": inserted_count,
        "skipped": {
            "duplicates": skipped_duplicates,
            "unsupported": skipped_unsupported,
            "empty": skipped_empty,
            "too_large": skipped_too_large,
            "parse_error": skipped_parse_error,
        },
    }
