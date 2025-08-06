from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash
from app.logic.ml_interface import clean_text, model, vectorizer, label_encoder
import uuid

router = APIRouter()

@router.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    parsed_resumes = []

    for file in files:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")

        contents = await file.read()
        resume_data = parse_resume_file(file.filename, contents)

        raw_text = resume_data.get("raw_text", "")
        if not raw_text.strip():
            continue

        # 🚫 Skip duplicates
        if await check_duplicate_cv_hash(raw_text):
            continue

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
        resume_data["name"] = resume_data.get("name") or "Unnamed Candidate"
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

        # ✅ Save to database
        await db.parsed_resumes.insert_one(resume_data)

        # ✅ Minimal safe fields for frontend
        parsed_resumes.append({
            "_id": resume_data["_id"],
            "name": resume_data["name"],
            "predicted_role": resume_data["predicted_role"],
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
        "resumes": parsed_resumes
    }
