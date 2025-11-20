# app/routers/upload_router.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import os
import re

from app.routers.auth_router import get_current_user  # ✅ auth required
from app.logic.resume_parser import parse_resume_file
from app.utils.mongo import db, compute_cv_hash, check_duplicate_cv_hash

# Classic ML classifier assets (best-effort, fully optional)
from app.logic.ml_interface import (
    clean_text,
    model,
    vectorizer,
    label_encoder,
    add_to_index,  # ✅ ANN: owner-scoped in-memory index (best-effort)
)

# Embeddings / ANN config (best-effort)
from app.logic.ml_interface import (
    embedding_model,      # should provide .encode(text) -> vector
    INDEX_TRUNCATE_CHARS, # int, for index text truncation
    MODEL_ID,             # embedding model/version identifier
)

# Normalizers for exact-match fields
from app.logic.normalize import (
    normalize_role,
    normalize_tokens,
    to_years_of_experience,
)

# numpy is optional; guard usage
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

router = APIRouter()

ALLOWED_EXTS = {".pdf", ".doc", ".docx"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# ✅ UI timing hints for the frontend (ms) — toaster should disappear ~3–4s
TOAST_HOLD_MS = 3500          # green toast ~3.5s
PROGRESS_HOLD_MS = 2000       # progress bar stays ~2s after 100%


def _round_to_half(n: float) -> float:
    try:
        return round(n * 2) / 2.0
    except Exception:
        return 0.0


def _experience_display(n: float) -> str:
    try:
        if n > 0 and n < 1:
            return "< 1 year"
        if n >= 1:
            r = int(round(n))
            return f"{r} year" + ("" if r == 1 else "s")
    except Exception:
        pass
    return "Not specified"


def _sanitize_for_mongo(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace obvious None values with safe defaults for known fields so Mongo
    doesn't get nulls that break the UI. Only touches a conservative set.
    """
    string_fields = [
        "name", "location", "company", "predicted_role", "category", "currentRole",
        "email", "phone", "filename", "resume_url", "legacy_ml_predicted_role",
    ]
    list_fields = ["skills", "resume.education", "resume.workHistory", "resume.projects"]

    for f in string_fields:
        if "." in f:
            parent, child = f.split(".", 1)
            if parent in doc and isinstance(doc[parent], dict) and doc[parent].get(child) is None:
                doc[parent][child] = ""
        else:
            if doc.get(f) is None:
                doc[f] = ""

    for f in list_fields:
        if "." in f:
            parent, child = f.split(".", 1)
            if parent in doc and isinstance(doc[parent], dict) and not isinstance(doc[parent].get(child), list):
                doc[parent][child] = []
        else:
            if not isinstance(doc.get(f), list):
                doc[f] = []

    # Numbers with sensible defaults (only for known numeric fields)
    for nf in ["experience", "total_experience_years", "years_of_experience", "experience_years", "yoe"]:
        v = doc.get(nf)
        try:
            doc[nf] = float(v) if v is not None else 0.0
        except Exception:
            doc[nf] = 0.0

    return doc


@router.post("/upload-resumes")
async def upload_resumes(
    files: List[UploadFile] = File(...),
    current_user=Depends(get_current_user),
):
    """
    Upload, parse, de-duplicate, normalize, persist (with embeddings when available),
    and warm the owner-scoped ANN index. Operates best-effort for embeddings—never fails
    the upload because of the ANN/embedding step.
    """
    # Input validation
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > 50:  # Reasonable limit to prevent abuse
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per upload")
    
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
        except Exception as e:
            # Log parse errors for debugging but don't fail the entire upload
            import logging
            logging.warning(f"Failed to parse resume {filename}: {e}")
            skipped_parse_error += 1
            skipped_files["parse_error"].append(filename)
            continue

        raw_text = (resume_data.get("raw_text") or "").strip()
        if not raw_text:
            skipped_empty += 1
            skipped_files["empty"].append(filename)
            continue

        # 🚫 Duplicate content (scoped to current user)
        if await check_duplicate_cv_hash(raw_text, owner_user_id=current_user.id):
            skipped_duplicates += 1
            skipped_files["duplicates"].append(filename)
            continue

        # --- Classic ML prediction (kept backward-compatible; guard if assets missing) ---
        cleaned = clean_text(raw_text)
        classic_ml_label: str = "Unknown"
        classic_ml_conf_pct: float = 0.0
        try:
            if vectorizer is not None and model is not None and label_encoder is not None:
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)
                # predict_proba may not exist; fallback safe
                _proba = getattr(model, "predict_proba", lambda x: [[0.0]])(features)[0]
                conf_val: float = float(max(_proba)) if isinstance(_proba, (list, tuple)) and _proba else 0.0
                classic_ml_label = label_encoder.inverse_transform(prediction)[0]
                classic_ml_conf_pct = round(conf_val * 100.0, 2)
        except Exception:
            classic_ml_label = "Unknown"
            classic_ml_conf_pct = 0.0

        # ---------- Construct the doc to insert ----------
        # Stable ID (avoids an extra DB roundtrip)
        _id = str(uuid.uuid4())

        # Ownership & de-dup hash
        content_hash = compute_cv_hash(raw_text)
        owner_id_str = str(current_user.id)

        # Experience canon (float years)
        def _to_float_years_upload(v: Any) -> float:
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

        # Normalized exact-match fields (idempotent; resume_parser already sets these—use setdefault)
        role_source = resume_data.get("currentRole") or resume_data.get("title") or resume_data.get("job_title")
        resume_data.setdefault("role_norm", normalize_role(role_source))
        resume_data.setdefault("skills_norm", normalize_tokens(resume_data.get("skills")))
        # projects_norm: from project names + tech tags if present
        if "projects_norm" not in resume_data:
            proj_items: List[str] = []
            _projects = (resume_data.get("resume") or {}).get("projects") or resume_data.get("projects") or []
            if isinstance(_projects, list):
                for p in _projects:
                    if isinstance(p, dict):
                        if p.get("name"):
                            proj_items.append(str(p["name"]))
                        if isinstance(p.get("tech"), list):
                            proj_items.extend([str(t) for t in p["tech"] if isinstance(t, str)])
                    elif isinstance(p, str):
                        proj_items.append(p)
            resume_data["projects_norm"] = normalize_tokens(proj_items)
        resume_data.setdefault("yoe_num", to_years_of_experience(numeric_years))

        # Canonical numeric aliases (kept for backward-compat; UI/pages may read any of these)
        resume_data["total_experience_years"] = numeric_years
        resume_data["years_of_experience"]    = numeric_years
        resume_data["experience_years"]       = numeric_years
        resume_data["yoe"]                    = numeric_years

        # Required metadata / fallbacks
        resume_data["_id"] = _id
        resume_data["content_hash"] = content_hash
        resume_data["ownerUserId"] = owner_id_str

        # ---------- Canonicalize & map role across synonyms ----------
        # Prefer parser-provided predicted_role; if missing/placeholder, fallback to parsed titles or classic ML.
        def _bad_role(s: Any) -> bool:
            t = (str(s or "").strip().lower())
            return (not t) or t in {"n/a", "na", "none", "-", "unknown", "not specified"}

        parser_role = (resume_data.get("currentRole") or resume_data.get("title") or resume_data.get("job_title") or "").strip()
        fallback_ml_role = (classic_ml_label or "").strip()

        if not resume_data.get("predicted_role") or _bad_role(resume_data.get("predicted_role")):
            canonical_role = parser_role if not _bad_role(parser_role) else (fallback_ml_role if not _bad_role(fallback_ml_role) else "")
            if canonical_role:
                resume_data["predicted_role"] = canonical_role
                resume_data["currentRole"] = canonical_role
                # refresh normalized role to align with canonical
                try:
                    resume_data["role_norm"] = normalize_role(canonical_role)
                except Exception:
                    pass

        # Do NOT clobber category if parser already provided it; only set if empty
        if not resume_data.get("category") or _bad_role(resume_data.get("category")):
            # fallback: align category to predicted_role if nothing better exists
            if resume_data.get("predicted_role") and not _bad_role(resume_data["predicted_role"]):
                resume_data["category"] = resume_data["predicted_role"]

        # Keep classic ML outputs (namespaced to avoid collisions)
        resume_data["legacy_ml_predicted_role"] = fallback_ml_role or "Unknown"
        resume_data["legacy_ml_confidence_pct"] = classic_ml_conf_pct

        # ---------- Role Prediction Confidence (stable, deterministic) ----------
        # Prefer role_confidence from resume_parser (0–1). If absent, derive from classic ML %.
        role_conf = 0.0
        try:
            if isinstance(resume_data.get("role_confidence"), (int, float)) and float(resume_data["role_confidence"]) > 0:
                role_conf = float(resume_data["role_confidence"])
            elif classic_ml_conf_pct > 0:
                role_conf = round(float(classic_ml_conf_pct) / 100.0, 3)
        except Exception:
            role_conf = 0.0

        resume_data["role_confidence"] = role_conf                      # 0–1
        resume_data["role_confidence_pct"] = round(role_conf * 100.0, 2)  # 0–100

        # For backward compatibility, normalize any existing ml_confidence to 0–1 and provide pct variant.
        if "ml_confidence" in resume_data:
            try:
                v = float(resume_data["ml_confidence"])
                resume_data["ml_confidence"] = v / 100.0 if v > 1.0 else v
            except Exception:
                resume_data["ml_confidence"] = role_conf
        else:
            resume_data["ml_confidence"] = role_conf
        resume_data["ml_confidence_pct"] = round(float(resume_data["ml_confidence"]) * 100.0, 2)

        # File / identity & basics
        resume_data["filename"] = filename
        resume_data["experience"] = resume_data.get("experience", numeric_years)
        resume_data["name"] = (resume_data.get("name") or "").strip() or "No Name"
        resume_data["location"] = resume_data.get("location") or "N/A"
        resume_data["raw_text"] = raw_text
        resume_data["email"] = resume_data.get("email") or None
        resume_data["phone"] = resume_data.get("phone") or None
        resume_data["skills"] = resume_data.get("skills") or []
        resume_data["resume_url"] = resume_data.get("resume_url", "")
        resume_data["company"] = resume_data.get("company") or "N/A"

        # Nested resume section (ensure presence)
        resume = resume_data.get("resume") or {}
        resume_data["resume"] = {
            "summary": resume.get("summary", ""),
            "education": resume.get("education", []),
            "workHistory": resume.get("workHistory", []),
            "projects": resume.get("projects", []),
            "filename": filename,
            "url": resume_data.get("resume_url", ""),
        }

        # Clean old transient fields if present
        for key in ["matchedSkills", "missingSkills", "score", "testScore", "rank", "filter_skills"]:
            resume_data.pop(key, None)

        # ---------- Build/use index blob (used for embedding + ANN warm) ----------
        try:
            # Prefer parser-provided index_blob if present
            index_blob = (resume_data.get("index_blob") or "").strip()
            if not index_blob:
                projects_blob = ""
                _p = resume_data["resume"]["projects"]
                if isinstance(_p, list):
                    parts: List[str] = []
                    for p in _p[:6]:
                        if isinstance(p, str):
                            parts.append(p)
                        elif isinstance(p, dict):
                            parts.append(str(p.get("name") or ""))
                            parts.append(str(p.get("description") or ""))
                    projects_blob = " ".join([x for x in parts if x])
                index_blob_raw = " ".join([
                    str(resume_data.get("predicted_role") or ""),
                    str(resume_data.get("name") or ""),
                    str(resume_data.get("location") or ""),
                    " ".join([s for s in (resume_data.get("skills") or []) if isinstance(s, str)]),
                    projects_blob,
                    str(resume_data.get("experience") or ""),
                    raw_text,
                ]).strip()
                index_blob = clean_text(index_blob_raw)[:int(INDEX_TRUNCATE_CHARS)]
        except Exception:
            index_blob = ""
        resume_data["index_blob"] = index_blob or None

        # ---------- Derived light fields (for UI queries & display) ----------
        resume_data["experience_rounded"] = _round_to_half(float(numeric_years))
        resume_data["experience_display"] = _experience_display(float(numeric_years))
        # location_norm: lowercased & trimmed (keep "n/a" as-is if already that)
        try:
            loc = (resume_data.get("location") or "").strip()
            resume_data["location_norm"] = loc.lower()
        except Exception:
            resume_data["location_norm"] = ""
        # search_blob: concise, sanitized text for quick text searches
        try:
            search_blob = " | ".join([
                str(resume_data.get("name") or ""),
                str(resume_data.get("predicted_role") or ""),
                " ".join([s for s in (resume_data.get("skills") or []) if isinstance(s, str)]),
                str(resume_data.get("location") or ""),
                str(resume_data.get("company") or ""),
            ])
            resume_data["search_blob"] = clean_text(search_blob)[:4096]
        except Exception:
            resume_data["search_blob"] = None

        # ---------- Embedding lifecycle fields (observable) ----------
        # If parser already provided an embedding, respect it and mark ready.
        existing_vec = resume_data.get("index_embedding")
        vec_to_store: Optional[List[float]] = None
        dim: Optional[int] = None

        if isinstance(existing_vec, list) and existing_vec:
            try:
                dim = len(existing_vec)
                resume_data["embedding_status"] = "ready"
                resume_data["embedding_model_version"] = str(MODEL_ID)
                resume_data["embedding_dim"] = dim
                resume_data["index_embedding"] = existing_vec
                resume_data["index_embedding_updated_at"] = datetime.now(timezone.utc)
                resume_data["ann_ready"] = True
                vec_to_store = existing_vec  # for ANN warm
            except Exception:
                existing_vec = None  # fall through to compute anew

        if vec_to_store is None:
            resume_data["embedding_status"] = "pending"
            resume_data["embedding_model_version"] = str(MODEL_ID)
            resume_data["embedding_dim"] = None
            resume_data["ann_ready"] = False

            # ---------- Compute embedding (best-effort, synchronous) ----------
            try:
                if embedding_model and index_blob:
                    vec = embedding_model.encode(index_blob, convert_to_tensor=False)
                    if np is not None:
                        vec_np = np.asarray(vec, dtype=np.float32)
                        nrm = float(np.linalg.norm(vec_np) + 1e-12)
                        if nrm > 0.0:
                            vec_np = (vec_np / nrm).astype(np.float32)
                        vec_to_store = vec_np.tolist()
                        dim = int(vec_np.shape[0])
                    else:
                        # numpy unavailable: store raw list, attempt simple L2 norm
                        lst = vec.tolist() if hasattr(vec, "tolist") else list(vec)  # type: ignore
                        try:
                            denom = (sum(x * x for x in lst) ** 0.5) or 1.0
                            lst = [float(x) / float(denom) for x in lst]
                        except Exception:
                            lst = [float(x) for x in lst]
                        vec_to_store = lst
                        dim = len(lst)
                    # Mark ready
                    resume_data["embedding_status"] = "ready"
                    resume_data["embedding_dim"] = dim
                    resume_data["index_embedding"] = vec_to_store
                    resume_data["index_embedding_updated_at"] = datetime.now(timezone.utc)
                    resume_data["ann_ready"] = True
            except Exception:
                # Embedding failed; keep pending/failed state
                resume_data["embedding_status"] = "failed"
                resume_data["ann_ready"] = False

        # ---------- Sanitize before persistence ----------
        resume_data = _sanitize_for_mongo(resume_data)

        # ---------- Persist once (single insert) ----------
        await db.parsed_resumes.insert_one(resume_data)
        inserted_count += 1

        # ---------- Warm in-memory ANN index (best-effort)
        # Prefer passing the precomputed embedding if the function supports it;
        # otherwise fall back to text_blob path.
        try:
            try:
                add_to_index(
                    owner_user_id=owner_id_str,
                    resume_id=_id,
                    text_blob=index_blob or "",
                    embedding=vec_to_store,   # type: ignore[arg-type]
                )
            except TypeError:
                # older signature without 'embedding'
                add_to_index(
                    owner_user_id=owner_id_str,
                    resume_id=_id,
                    text_blob=index_blob or "",
                )
        except Exception:
            # Never fail the request due to ANN warm-up
            pass

        # ---------- Safe preview for UI ----------
        parsed_resumes.append({
            "_id": resume_data["_id"],
            "name": resume_data["name"],
            "predicted_role": resume_data.get("predicted_role"),
            "category": resume_data.get("category"),
            "experience": resume_data["experience"],
            "location": resume_data["location"],
            # For cards: show Role Prediction Confidence as percent (stable)
            "confidence": resume_data.get("role_confidence_pct", 0.0),
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
        # ➕ UI hints the frontend will respect (toaster auto-dismiss in ~3–4s)
        "ui": {
            "toast": {
                "type": "success" if total_skipped == 0 else "warning",
                "text": toast_text,
                "holdMs": TOAST_HOLD_MS,
            },
            "progressHoldMs": PROGRESS_HOLD_MS,
        },
    }
