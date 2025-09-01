from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List, Dict, Any
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash
from app.logic.ml_interface import (
    clean_text,
    model,
    vectorizer,
    label_encoder,
    add_to_index,  # ✅ ANN: add newly inserted resume to owner-scoped in-memory index
)
from app.routers.auth_router import get_current_user  # ✅ auth required
import uuid
import os
import re  # ✅ needed for experience normalization

# ✅ NEW imports for embedding persistence & timestamps
from app.logic.ml_interface import embedding_model, INDEX_TRUNCATE_CHARS, MODEL_ID
from datetime import datetime, timezone
try:
    import numpy as np
except Exception:
    np = None  # degrade gracefully if numpy unavailable

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
    parsed_resumes: List[Dict[str, Any]] = []

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

        # --- ML prediction (kept backward-compatible; guard if assets missing) ---
        cleaned = clean_text(raw_text)
        predicted_label: str = "Unknown"
        confidence_pct: float = 0.0
        try:
            if vectorizer is not None and model is not None and label_encoder is not None:
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)
                confidence = getattr(model, "predict_proba", lambda x: [[0.0]])(features)[0]
                if isinstance(confidence, (list, tuple)) and len(confidence) > 0:
                    confidence = float(max(confidence))
                else:
                    confidence = 0.0
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                confidence_pct = round(float(confidence) * 100.0, 2)
        except Exception:
            # If classic ML assets fail, fallback to Unknown gracefully
            predicted_label = "Unknown"
            confidence_pct = 0.0

        # ✅ Required metadata
        resume_data["_id"] = str(uuid.uuid4())
        resume_data["content_hash"] = compute_cv_hash(raw_text)
        resume_data["predicted_role"] = predicted_label
        resume_data["category"] = predicted_label
        resume_data["confidence"] = confidence_pct
        resume_data["filename"] = filename
        resume_data["experience"] = resume_data.get("experience", 0)

        # 🔧 Canonical numeric experience fields for DB filtering & scoring
        def _to_float_years_upload(v):
            try:
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v)
                m = re.search(r"(\d+(?:\.\d+)?)", s)
                return float(m.group(1)) if m else 0.0
            except Exception:
                return 0.0

        exp_candidates = [
            resume_data.get("total_experience_years"),
            resume_data.get("years_of_experience"),
            resume_data.get("experience_years"),
            resume_data.get("yoe"),
            resume_data.get("experience"),
        ]

        numeric_years = 0.0
        for c in exp_candidates:
            if c is not None and str(c).strip() != "":
                numeric_years = _to_float_years_upload(c)
                break

        # Set all aliases so DB-side $gte works reliably
        resume_data["total_experience_years"] = numeric_years
        resume_data["years_of_experience"]    = numeric_years
        resume_data["experience_years"]       = numeric_years
        resume_data["yoe"]                    = numeric_years

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

        # ✅ Insert into DB
        await db.parsed_resumes.insert_one(resume_data)
        inserted_count += 1

        # ✅ ANN: Update in-memory owner-scoped vector index immediately (best-effort, non-fatal)
        try:
            # Compact blob similar to backend indexer (clean/trim happens inside add_to_index)
            projects_blob = ""
            if isinstance(resume.get("projects"), list):
                parts = []
                for p in resume["projects"][:6]:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        parts.append(str(p.get("name") or ""))
                        parts.append(str(p.get("description") or ""))
                projects_blob = " ".join([x for x in parts if x])
            index_blob = " ".join([
                str(resume_data.get("predicted_role") or ""),
                str(resume_data.get("name") or ""),
                str(resume_data.get("location") or ""),
                " ".join([s for s in (resume_data.get("skills") or []) if isinstance(s, str)]),
                projects_blob,
                str(resume_data.get("experience") or ""),
                raw_text,
            ]).strip()
            add_to_index(
                owner_user_id=str(current_user.id),
                resume_id=resume_data["_id"],
                text_blob=index_blob,
            )
        except Exception:
            # Index update is best-effort; ignore failures
            pass

        # ✅ NEW: Persist index embedding + index_blob for fast ANN (best-effort, non-fatal)
        try:
            # Clean + truncate for encoder, consistent with ANN builder
            blob_for_emb = clean_text(index_blob)[:int(INDEX_TRUNCATE_CHARS)]
            vec = embedding_model.encode(blob_for_emb, convert_to_tensor=False)

            # Normalize & ensure float32 if numpy is available
            if np is not None:
                try:
                    vec_np = np.asarray(vec, dtype=np.float32)
                    nrm = np.linalg.norm(vec_np) + 1e-12
                    vec_np = (vec_np / nrm).astype(np.float32)
                    vec_to_store = vec_np.tolist()
                    dim = int(vec_np.shape[0])
                except Exception:
                    # fallback: store raw list
                    vec_to_store = [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
                    dim = len(vec_to_store)
            else:
                # numpy not available: store as-is
                vec_to_store = [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
                dim = len(vec_to_store)

            await db.parsed_resumes.update_one(
                {"_id": resume_data["_id"]},
                {"$set": {
                    "index_blob": blob_for_emb,
                    "index_embedding": vec_to_store,
                    "index_embedding_dim": dim,
                    "index_embedding_model": str(MODEL_ID),
                    "ann_ready": True,
                    "index_embedding_updated_at": datetime.now(timezone.utc),
                }}
            )
        except Exception:
            # Best-effort only; do not fail the upload on embedding issues
            pass

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
