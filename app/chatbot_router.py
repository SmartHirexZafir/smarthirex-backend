# ✅ File: app/chatbot_router.py

from fastapi import APIRouter, Request, Depends
from datetime import datetime, timezone
from typing import Any, Optional, Dict, List

from bson import ObjectId
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db
from app.routers.auth_router import get_current_user  # ✅ enforce auth
from app.logic.normalize import normalize_prompt, extract_min_years  # ✅ correct imports
import re

router = APIRouter()


def _to_object_id(val: Any) -> Optional[ObjectId]:
    """
    Safely convert unknown id shape to ObjectId.
    Returns None if conversion not possible.
    """
    if isinstance(val, ObjectId):
        return val
    if isinstance(val, dict) and "$oid" in val:
        try:
            return ObjectId(val["$oid"])
        except Exception:
            return None
    if isinstance(val, (bytes, bytearray)):
        try:
            return ObjectId(val)
        except Exception:
            return None
    if isinstance(val, str):
        try:
            return ObjectId(val)
        except Exception:
            return None
    return None


def _candidate_text_blob(c: Dict[str, Any]) -> str:
    """
    Build a lowercase searchable haystack from candidate fields.
    """
    parts: List[str] = []

    def push(v: Any):
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                push(x)
        elif isinstance(v, dict):
            for x in v.values():
                push(x)
        else:
            sv = str(v).strip()
            if sv:
                parts.append(sv)

    push(c.get("name"))
    push(c.get("predicted_role"))
    push(c.get("category"))
    push(c.get("currentRole"))
    push(c.get("skills"))
    push(c.get("summary"))
    # nested resume fields
    r = c.get("resume") or {}
    push(r.get("summary"))
    push(r.get("projects"))
    push(r.get("workHistory"))
    return " • ".join(parts).lower()


def _candidate_title_text(c: Dict[str, Any]) -> str:
    """Collect title-like fields for strict role checks."""
    r = c.get("resume") or {}
    parts = [
        c.get("predicted_role"),
        c.get("category"),
        c.get("currentRole"),
        r.get("title"),
        r.get("job_title"),
        r.get("current_title"),
        r.get("headline"),
    ]
    return " ".join(str(x) for x in parts if x).lower()


def _role_matches(wanted_role: Optional[str], c: Dict[str, Any]) -> bool:
    """
    Returns True if:
      - no role requested, or
      - requested role (or its suffix word) appears in candidate title text
    """
    if not wanted_role:
        return True
    want = wanted_role.strip().lower()
    title_blob = _candidate_title_text(c)

    # direct substring
    if want and want in title_blob:
        return True

    # suffix-based (designer|developer|engineer|scientist|analyst|manager|architect)
    # and presence of a leading keyword (e.g., "web", "data", "ui/ux")
    m = re.search(
        r"\b([a-z][a-z ]*?)\s+(designer|developer|engineer|scientist|analyst|manager|architect)\b",
        want,
    )
    if m:
        lead = m.group(1).strip()
        suffix = m.group(2).strip()
        if suffix in title_blob and (not lead or lead in title_blob):
            return True

    # light synonyms for design roles
    if "designer" in want or "design" in want:
        if any(tok in title_blob for tok in ["designer", "design", "ui", "ux"]):
            return True

    return False


def _candidate_years(c: Dict[str, Any]) -> float:
    """
    Try to convert experience into a number of years.
    """
    exp = c.get("experience")
    if isinstance(exp, (int, float)):
        return float(exp)
    if isinstance(exp, str):
        m = re.search(r"(\d+(?:\.\d+)?)", exp)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
    return 0.0


async def _owner_has_resume(owner_id: str) -> bool:
    return await db.parsed_resumes.count_documents({"ownerUserId": owner_id}) > 0


async def _owner_preview_filter(items: List[Dict[str, Any]], owner_id: str) -> List[Dict[str, Any]]:
    """
    Keep only previews that belong to the current owner.
    Support both string UUID _id and ObjectId _id stored in DB.
    """
    filtered: List[Dict[str, Any]] = []
    for item in items or []:
        cid_any = item.get("_id") or item.get("id") or item.get("resume_id")

        # Prepare a query that tries multiple possible id shapes
        or_conditions = []
        # string ID path
        if isinstance(cid_any, str):
            or_conditions.append({"_id": cid_any, "ownerUserId": owner_id})

        # objectId path (from raw or {$oid})
        oid = _to_object_id(cid_any)
        if oid:
            or_conditions.append({"_id": oid, "ownerUserId": owner_id})

        if not or_conditions:
            # if we can't form any id condition, skip quietly
            continue

        doc = await db.parsed_resumes.find_one(
            {"$or": or_conditions},
            {"_id": 1},
        )
        if doc:
            # ensure safe fallbacks for frontend card
            if not (item.get("name") and str(item.get("name")).strip()):
                item["name"] = "No Name"
            if not isinstance(item.get("skills"), list):
                item["skills"] = []
            filtered.append(item)
    return filtered


async def _fallback_filter_for_owner(
    owner_id: str,
    prompt_like: str,
    provided_keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    If primary pipeline returns no results, try a lightweight fuzzy filter
    over the owner's resumes to avoid false 'no results'.
    - Adds a strict *role/title* check so irrelevant roles are not returned.
    """
    # 🔧 use imported normalize helpers (fixed names)
    norm = normalize_prompt(prompt_like)
    keywords = provided_keywords or norm["keywords"]
    min_years = extract_min_years(norm["normalized_prompt"])

    # parse a concrete job title (e.g. 'web designer', 'data scientist')
    parsed = parse_prompt(prompt_like) or {}
    wanted_role = parsed.get("job_title")

    # fetch a reasonable slice (you can index later for scale)
    cursor = db.parsed_resumes.find(
        {"ownerUserId": owner_id},
        {
            "_id": 1,
            "name": 1,
            "predicted_role": 1,
            "category": 1,
            "experience": 1,
            "location": 1,
            "confidence": 1,
            "email": 1,
            "phone": 1,
            "skills": 1,
            "resume_url": 1,
            "filename": 1,
            "currentRole": 1,
            "resume": 1,
            "semantic_score": 1,
            "final_score": 1,
        },
    )

    docs = await cursor.to_list(length=1000)

    scored: List[Dict[str, Any]] = []
    for c in docs:
        # 🚫 discard if role/title clearly doesn't match
        if not _role_matches(wanted_role, c):
            continue

        hay = _candidate_text_blob(c)
        score = 0.0

        for k in keywords:
            if k in hay:
                score += 10
            # tiny preference for role-ish tokens
            if re.search(r"(developer|engineer|scientist|designer|manager|architect|analyst)", k):
                score += 5

        yrs = _candidate_years(c)
        if min_years is not None and yrs >= min_years:
            score += 15

        # leverage existing scores if present
        if isinstance(c.get("semantic_score"), (int, float)):
            score += float(c["semantic_score"])
        if isinstance(c.get("final_score"), (int, float)):
            score += float(c["final_score"])

        if score > 0:
            # map to preview shape
            preview = {
                "_id": c.get("_id"),
                "name": (str(c.get("name") or "").strip() or "No Name"),
                "predicted_role": c.get("predicted_role"),
                "category": c.get("category"),
                "experience": c.get("experience"),
                "location": c.get("location"),
                "confidence": c.get("confidence"),
                "email": c.get("email"),
                "phone": c.get("phone"),
                "skills": c.get("skills") or [],
                "resume_url": c.get("resume_url"),
                "filename": c.get("filename"),
                "score_type": "Fallback Match",
                "semantic_score": c.get("semantic_score"),
                "final_score": c.get("final_score"),
                # Mark as close so UI can show the right banner
                "is_strict_match": False,
                "match_type": "close",
            }
            scored.append({"preview": preview, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [x["preview"] for x in scored[:100]]  # return top-N


def _summarize_matches(preview: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Derive exact/close counts and booleans for UI messaging.
    Works even if items don't carry the new flags (assumes exact in that case).
    """
    strict_count = 0
    close_count = 0
    for p in preview or []:
        if p.get("is_strict_match") is True or p.get("match_type") == "exact":
            strict_count += 1
        elif p.get("is_strict_match") is False or p.get("match_type") == "close":
            close_count += 1
        else:
            # If flags are missing, treat as strict by default
            strict_count += 1
    total = len(preview or [])
    return {
        "strictCount": strict_count,
        "closeCount": close_count,
        "total": total,
        "hasExact": strict_count > 0,
        "hasClose": close_count > 0 or (total > 0 and strict_count < total),
    }


@router.post("/query")
async def handle_chatbot_query(
    request: Request,
    current_user=Depends(get_current_user),
):
    """
    Receives a natural language prompt from user,
    identifies intent, processes it, and responds accordingly.
    """
    data = await request.json()
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return {"reply": "Prompt is empty.", "resumes_preview": []}

    owner_id = str(current_user.id)

    # ✅ Early guard: if this user has no CVs, short-circuit with a clear flag
    if not await _owner_has_resume(owner_id):
        return {
            "reply": "You haven't uploaded any resumes yet.",
            "message": "no_cvs_uploaded",
            "no_cvs_uploaded": True,
            "no_results": False,
            "resumes_preview": [],
            "matchMeta": {"strictCount": 0, "closeCount": 0, "total": 0, "hasExact": False, "hasClose": False},
        }

    # --- normalization (use client-provided if available, else compute)
    normalized_prompt = (data.get("normalized_prompt") or "").strip()
    keywords = data.get("keywords")
    if not normalized_prompt or not isinstance(keywords, list):
        norm = normalize_prompt(prompt)  # 🔧 correct function
        normalized_prompt = norm["normalized_prompt"]
        keywords = norm["keywords"]

    # Step 1: Parse intent (enrich parsed with normalized info for downstream)
    parsed = parse_prompt(prompt) or {}
    parsed.setdefault("normalized_prompt", normalized_prompt)
    parsed.setdefault("keywords", keywords)
    parsed.setdefault("ownerUserId", owner_id)  # 👈 pass owner for downstream filtering

    # Step 2: Build response (existing pipeline)
    response = await build_response(parsed)  # keep as-is

    # ✅ Safety post-filter: keep only resumes owned by current user
    raw_preview = response.get("resumes_preview", []) or []
    filtered_preview = await _owner_preview_filter(raw_preview, owner_id)

    # ✅ Server-side fallback fuzzy filter if nothing left (now role-aware)
    if not filtered_preview:
        filtered_preview = await _fallback_filter_for_owner(
            owner_id=owner_id,
            prompt_like=normalized_prompt or prompt,
            provided_keywords=keywords,
        )

    # Summarize matches for UI messaging
    match_meta = _summarize_matches(filtered_preview)
    no_results = match_meta["total"] == 0  # 👈 add this flag

    # Step 3: Log analytics (scoped)
    await db.chat_queries.insert_one(
        {
            "prompt": prompt,
            "parsed": parsed,
            "response_preview_count": len(filtered_preview),
            "timestamp": datetime.now(timezone.utc),
            "ownerUserId": owner_id,
            "matchMeta": match_meta,
        }
    )

    # Step 4: Save history for filter intents
    if (parsed.get("intent") == "filter_cv") and filtered_preview:
        timestamp_raw = datetime.now(timezone.utc)
        timestamp_display = datetime.now().strftime("%B %d, %Y – %I:%M %p")
        await db.search_history.insert_one(
            {
                "prompt": prompt,
                "parsed": parsed,
                "timestamp_raw": timestamp_raw,
                "timestamp_display": timestamp_display,
                "totalMatches": len(filtered_preview),
                "candidates": filtered_preview,
                "ownerUserId": owner_id,
                "matchMeta": match_meta,
            }
        )

    # Step 5: Context-aware reply text for UI banners
    count = match_meta["total"]
    reply_text = f"Showing {count} result{'s' if count != 1 else ''} for your query."
    if count == 0:
        reply_text = "Showing 0 results for your query."

    return {
        "reply": reply_text,                     # for the chat bubble
        "resumes_preview": filtered_preview,     # cards
        "normalized_prompt": normalized_prompt,  # extra context for UI
        "keywords": keywords,                    # extra context for UI
        "matchMeta": match_meta,                 # counts for banners
        "no_results": no_results,                # 👈 NEW flag for UI (hide analyzing + 3s toast)
        "no_cvs_uploaded": False,
        # convenience for candidate filter header
        "ui": {
            "primaryMessage": reply_text,        # "Showing N results for your query."
            "query": prompt,                     # original query text (to print under header)
        },
    }
