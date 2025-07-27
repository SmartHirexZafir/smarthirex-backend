# ✅ File: app/routers/upload_router.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db
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
        resume_data["_id"] = str(uuid.uuid4())

        # ✅ ML Scoring Enhancement
        raw_text = resume_data.get("raw_text", "")
        cleaned = clean_text(raw_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        skills = set(s.lower() for s in resume_data.get("skills", []))
        required_skills = skills  # use extracted skills as baseline

        resume_data["predicted_role"] = predicted_label
        resume_data["score"] = 100
        resume_data["matchedSkills"] = list(skills)
        resume_data["missingSkills"] = []
        resume_data["testScore"] = 0
        resume_data["rank"] = 0

        await db.parsed_resumes.insert_one(resume_data)
        parsed_resumes.append(resume_data)

    return {"message": "Resumes processed successfully", "resumes": parsed_resumes}
