# ✅ File: app/routers/upload_router.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db
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
        resume_data["_id"] = str(uuid.uuid4())  # unique ID for each resume
        await db.parsed_resumes.insert_one(resume_data)
        parsed_resumes.append(resume_data)

    return {"message": "Resumes processed successfully", "resumes": parsed_resumes}
