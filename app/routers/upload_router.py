from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List, Dict
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash
from app.logic.ml_interface import clean_text, model, vectorizer, label_encoder
from app.routers.auth_router import get_current_user  # ✅ auth required
import uuid
import os

router = APIRouter()

ALLOWED_EXTS = {".pdf", ".doc", ".docx"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# ✅ UI timing hints for the frontend (ms)
TOAST_HOLD_MS = 2500          # green toast ~2–3s
PROGRESS_HOLD_MS = 2000       # progress bar stays ~2s after 100%


@router.post("/upload-resumes")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    current_user = Depends(get_current_user),  # ✅ enforce login/ownership
):
    parsed_resumes: List[Dict] = []

    received_count = len(files)
    inserted_count = 0
    skipped_duplicates = 0
    skipped_unsupported = 0
    skipped_empty = 0
    skipped_too_large = 0
    skipped_parse_error = 0

    # Per-reason filename lists for precise UI feedback
    skipped_files = {
        "duplicates": [],
        "unsupported": [],
        "empty": [],
        "too_large": [],
        "parse_error": [],
    }

    for file in files:
        filename = file.filename or "unnamed"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTS:
            skipped_unsupported += 1
            skipped_files["unsupported"].append(filename)
            continue

        contents = await file.read()
        if not contents:
            skipped_empty += 1
            skipped_files["empty"].append(filename)
            continue

        if len(contents) > MAX_SIZE_BYTES:
            skipped_too_large += 1
            skipped_files["too_large"].append(filename)
            continue

        # Parse safely
        try:
            resume_data = parse_resume_file(filename, contents)
        except Exception:
            skipped_parse_error += 1
            skipped_files["parse_error"].append(filename)
            continue

        raw_text = resume_data.get("raw_text", "") or ""
        if not raw_text.strip():
            skipped_empty += 1
            skipped_files["empty"].append(filename)
            continue

        # 🚫 Duplicate content (scoped to current user)
        if await check_duplicate_cv_hash(raw_text, owner_user_id=current_user.id):
            skipped_duplicates += 1
            skipped_files["duplicates"].append(filename)
            continue

        # --- ML prediction (unchanged) ---
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)
        confidence = model.predict_proba(features)[0].max()
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # ✅ Required metadata
        resume_data["_id"] = str(uuid.uuid4())
        resume_data["content_hash"] = compute_cv_hash(raw_text)
        resume_data["predicted_role"] = predicted_label
        resume_data["category"] = predicted_label
        resume_data["confidence"] = round(confidence * 100, 2)
        resume_data["filename"] = filename
        resume_data["experience"] = resume_data.get("experience", 0)

        # Fallbacks & profile fields
        resume_data["name"] = (resume_data.get("name") or "").strip() or "No Name"
        resume_data["location"] = resume_data.get("location") or "N/A"
        resume_data["raw_text"] = raw_text
        resume_data["email"] = resume_data.get("email") or None
        resume_data["phone"] = resume_data.get("phone") or None
        resume_data["skills"] = resume_data.get("skills") or []
        resume_data["resume_url"] = resume_data.get("resume_url", "")
        resume_data["currentRole"] = resume_data.get("currentRole") or "N/A"
        resume_data["company"] = resume_data.get("company") or "N/A"

        # Nested resume section
        resume_data["resume"] = resume_data.get("resume", {})
        resume = resume_data["resume"]
        resume["summary"] = resume.get("summary", "")
        resume["education"] = resume.get("education", [])
        resume["workHistory"] = resume.get("workHistory", [])
        resume["projects"] = resume.get("projects", [])
        resume["filename"] = filename
        resume["url"] = resume_data.get("resume_url", "")

        # Clean old transient fields if present
        for key in ["matchedSkills", "missingSkills", "score", "testScore", "rank", "filter_skills"]:
            resume_data.pop(key, None)

        # ✅ Ownership
        resume_data["ownerUserId"] = str(current_user.id)

        # ✅ Insert
        await db.parsed_resumes.insert_one(resume_data)
        inserted_count += 1

        # Safe preview for UI
        parsed_resumes.append({
            "_id": resume_data["_id"],
            "name": resume_data["name"],
            "predicted_role": resume_data["predicted_role"],
            "category": resume_data["category"],
            "experience": resume_data["experience"],
            "location": resume_data["location"],
            "confidence": resume_data["confidence"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "skills": resume_data["skills"],
            "resume_url": resume_data["resume_url"],
            "filename": resume_data["filename"],
        })

    total_skipped = (
        skipped_duplicates + skipped_unsupported + skipped_empty + skipped_too_large + skipped_parse_error
    )

    toast_text = (
        f"✅ Uploaded {inserted_count} of {received_count}"
        f" · Duplicates: {skipped_duplicates}"
        f" · Unsupported: {skipped_unsupported}"
        f" · Empty: {skipped_empty}"
        f" · Too large: {skipped_too_large}"
        f" · Parse errors: {skipped_parse_error}"
    )

    return {
        "message": "Resumes processed successfully",
        "resumes": parsed_resumes,
        "received": received_count,
        "inserted": inserted_count,
        "skipped": {
            "total": total_skipped,
            "duplicates": skipped_duplicates,
            "unsupported": skipped_unsupported,
            "empty": skipped_empty,
            "too_large": skipped_too_large,
            "parse_error": skipped_parse_error,
            "filenames": skipped_files,  # per-reason filename lists
        },
        # ➕ UI hints the frontend will respect
        "ui": {
            "toast": {
                "type": "success" if total_skipped == 0 else "warning",
                "text": toast_text,
                "holdMs": TOAST_HOLD_MS,
            },
            "progressHoldMs": PROGRESS_HOLD_MS,
        },
    }
