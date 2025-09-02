# ✅ File: app/chatbot_router.py

from fastapi import APIRouter, Request, Depends
from datetime import datetime, timezone
from typing import Any, Optional, Dict, List, Iterable

from bson import ObjectId
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db
from app.routers.auth_router import get_current_user  # ✅ enforce auth
from app.logic.normalize import normalize_prompt, extract_min_years  # ✅ correct imports
import re

router = APIRouter()

# --- Allowed filters (canonical keys) ---------------------------------------
_ALLOWED_FILTERS = {
    "role": "role",                # Job Role
    "job_role": "role",
    "jobrole": "role",
    "skills": "skills",            # Skills
    "location": "location",        # Location
    "projects": "projects",        # Projects
    "experience": "experience",    # Experience
    "cv": "cv",                    # CV Content Matching
    "cv_content": "cv",
    "cv_content_matching": "cv",
}

# Minimal baseline skills for routing-level filtering (aligned with backend list)
_BASELINE_SKILLS = set([
    "react", "aws", "node", "excel", "django", "figma",
    "pandas", "tensorflow", "keras", "java", "python",
    "pytorch", "spark", "sql", "scikit", "mlflow", "docker",
    "kubernetes", "typescript", "next", "nextjs", "next.js",
    "powerpoint", "flora", "html", "css"
])

def _canon_filters(raw_filters: Any) -> List[str]:
    """
    Canonicalize selected filter keys and preserve the original order.
    """
    if not isinstance(raw_filters, (list, tuple)):
        return []
    out: List[str] = []
    seen = set()
    for f in raw_filters:
        k = str(f or "").strip().lower()
        k = _ALLOWED_FILTERS.get(k, "")
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


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

    # nested resume fields (if present in preview)
    r = c.get("resume") or {}
    push(r.get("summary"))
    push(r.get("projects"))
    push(r.get("workHistory"))

    # raw_text (from ml layer) if passed through
    push(c.get("raw_text"))

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
    Try to convert experience into a number of years (handles multiple aliases).
    """
    for k in ("experience", "total_experience_years", "years_of_experience", "experience_years", "yoe"):
        exp = c.get(k)
        if isinstance(exp, (int, float)):
            return float(exp)
        if isinstance(exp, str):
            m = re.search(r"(\d+(?:\.\d+)?)", exp)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    return 0.0


async def _owner_has_resume(owner_id: str) -> bool:
    return await db.parsed_resumes.count_documents({"ownerUserId": owner_id}) > 0


async def _owner_preview_filter(items: List[Dict[str, Any]], owner_id: str) -> List[Dict[str, Any]]:
    """
    Keep only previews that belong to the current owner.
    Support both string UUID _id and ObjectId _id stored in DB.

    🔧 Performance fix: resolve ownership in BULK (avoids N+1 DB lookups).
    """
    if not items:
        return []

    str_ids: List[str] = []
    obj_ids: List[ObjectId] = []

    for item in items:
        cid_any = item.get("_id") or item.get("id") or item.get("resume_id")
        if isinstance(cid_any, str):
            str_ids.append(cid_any)
        oid = _to_object_id(cid_any)
        if oid:
            obj_ids.append(oid)

    # Build $or with type-separated _id lists (Mongo can't mix types in one $in reliably)
    or_conditions: List[Dict[str, Any]] = []
    if str_ids:
        or_conditions.append({"_id": {"$in": str_ids}, "ownerUserId": owner_id})
    if obj_ids:
        or_conditions.append({"_id": {"$in": obj_ids}, "ownerUserId": owner_id})

    if not or_conditions:
        return []

    cursor = db.parsed_resumes.find({"$or": or_conditions}, {"_id": 1})
    owned = await cursor.to_list(length=10_000)
    owned_ids: set = set()
    for d in owned:
        _id = d.get("_id")
        owned_ids.add(str(_id))

    filtered: List[Dict[str, Any]] = []
    for item in items:
        cid_any = item.get("_id") or item.get("id") or item.get("resume_id")
        if str(cid_any) in owned_ids:
            if not (item.get("name") and str(item.get("name")).strip()):
                item["name"] = "No Name"
            if not isinstance(item.get("skills"), list):
                item["skills"] = []
            filtered.append(item)
    return filtered


# ------------------------ server-side SEQUENTIAL FILTERS ---------------------

def _extract_location_from_prompt(prompt_like: str) -> Optional[str]:
    """
    Very lightweight location extractor: looks for 'in <place>'.
    """
    p = " " + (prompt_like or "").lower() + " "
    m = re.search(r"\bin\s+([a-z][a-z .,'-]{1,60})", p)
    if not m:
        return None
    loc = m.group(1).strip()
    # trim trailing punctuation
    loc = re.sub(r"[.,;:!?]+$", "", loc)
    return loc if loc else None


def _filter_by_role(cands: Iterable[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    wanted_role = (parse_prompt(prompt) or {}).get("job_title")
    return [c for c in cands if _role_matches(wanted_role, c)]


def _filter_by_experience(cands: Iterable[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    # treat prompt as ">= N years" when a single value is found
    min_years = extract_min_years(prompt)  # None if not present
    if min_years is None:
        return list(cands)
    out = []
    for c in cands:
        yrs = _candidate_years(c)
        if yrs >= float(min_years):
            out.append(c)
    return out


def _filter_by_skills(cands: Iterable[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    # only use plausible skill tokens from prompt
    focus = [k for k in (keywords or []) if k in _BASELINE_SKILLS]
    if not focus:
        return list(cands)
    focus_set = set(focus)
    out: List[Dict[str, Any]] = []
    for c in cands:
        skills = c.get("skills") or []
        skills_lc = set(str(s).strip().lower() for s in skills if s)
        if skills_lc & focus_set:
            out.append(c)
    return out


def _filter_by_location(cands: Iterable[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    loc = _extract_location_from_prompt(prompt)
    if not loc:
        return list(cands)
    out: List[Dict[str, Any]] = []
    for c in cands:
        c_loc = str(c.get("location") or "").lower()
        if c_loc and loc in c_loc:
            out.append(c)
    return out


def _filter_by_projects(cands: Iterable[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Project-category-only filter.
    Looks ONLY at 'projects' field (strings or dicts with name/description).
    """
    if not keywords:
        return list(cands)
    keys = [k.strip().lower() for k in keywords if k and isinstance(k, str)]
    out: List[Dict[str, Any]] = []
    for c in cands:
        txt = ""
        projs = c.get("projects", [])
        if isinstance(projs, list):
            for p in projs:
                if isinstance(p, str):
                    txt += " " + p
                elif isinstance(p, dict):
                    txt += " " + str(p.get("name", "")) + " " + str(p.get("description", ""))
        txt = txt.lower()
        if txt and any(k in txt for k in keys):
            out.append(c)
    return out


def _filter_by_cv_content(cands: Iterable[Dict[str, Any]], prompt: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """
    When CV content is selected, only match prompt keywords / substring against raw_text-like content.
    We keep this conservative (must match at least one keyword or prompt substring).
    """
    p = (prompt or "").lower().strip()
    keys = [k for k in (keywords or []) if k and len(k) > 1]
    out: List[Dict[str, Any]] = []
    for c in cands:
        blob = (
            str(c.get("raw_text") or "")
            + " "
            + str((c.get("resume") or {}).get("summary") or "")
            + " "
            + str((c.get("resume") or {}).get("workHistory") or "")
        ).lower()
        if not blob:
            continue
        if p and p in blob:
            out.append(c)
            continue
        if keys and any(k in blob for k in keys):
            out.append(c)
    return out


def _apply_selected_filters_in_order(
    items: List[Dict[str, Any]],
    selected: List[str],
    prompt_like: str,
    keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    Sequentially apply the selected filters in the given order.
    Each step reduces the working set → faster & stricter.
    """
    working = list(items)
    if not working or not selected:
        return working

    for f in selected:
        if f == "role":
            working = _filter_by_role(working, prompt_like)
        elif f == "experience":
            working = _filter_by_experience(working, prompt_like)
        elif f == "skills":
            working = _filter_by_skills(working, keywords)
        elif f == "location":
            working = _filter_by_location(working, prompt_like)
        elif f == "projects":
            working = _filter_by_projects(working, keywords)
        elif f == "cv":
            working = _filter_by_cv_content(working, prompt_like, keywords)
        # if list becomes empty early, break
        if not working:
            break
    return working


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
    try:
        data = await request.json()
    except Exception:
        return {
            "reply": "Invalid request payload.",
            "resumes_preview": [],
            "matchMeta": {"strictCount": 0, "closeCount": 0, "total": 0, "hasExact": False, "hasClose": False},
            "no_results": True,
            "no_cvs_uploaded": False,
            "ui": {"primaryMessage": "Invalid request payload.", "query": ""},
        }

    prompt = (data.get("prompt") or "").strip()
    selected_filters_raw = data.get("selected_filters") or data.get("filters") or data.get("selectedFilters") or []
    selected_filters = _canon_filters(selected_filters_raw)

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
    # 👇 NEW: include selected filters (order preserved)
    parsed["selectedFilters"] = selected_filters

    # Step 2: Build response (existing pipeline)
    try:
        response = await build_response(parsed)  # keep as-is
    except Exception:
        # Fail-safe: return an empty, UI-safe response so loader stops
        return {
            "reply": "Sorry, something went wrong while processing your query.",
            "resumes_preview": [],
            "normalized_prompt": normalized_prompt,
            "keywords": keywords,
            "matchMeta": {"strictCount": 0, "closeCount": 0, "total": 0, "hasExact": False, "hasClose": False},
            "no_results": True,
            "no_cvs_uploaded": False,
            "ui": {"primaryMessage": "Sorry, something went wrong while processing your query.", "query": prompt},
        }

    # ✅ Safety post-filter: keep only resumes owned by current user
    raw_preview = response.get("resumes_preview", []) or []
    filtered_preview = await _owner_preview_filter(raw_preview, owner_id)

    # ✅ NEW: Semantic fallback (owner-scoped) so cards get semantic_score, related_roles, role etc.
    # If still empty after the first pass, use get_semantic_matches instead of the old fuzzy DB scan.
    if not filtered_preview:
        try:
            from app.logic.ml_interface import get_semantic_matches  # lazy import to avoid cycles
            sem_candidates = await get_semantic_matches(
                normalized_prompt or prompt,
                owner_user_id=owner_id,
                normalized_prompt=normalized_prompt,
                keywords=keywords,
            )
        except Exception:
            sem_candidates = []
        # keep ownership safety
        filtered_preview = await _owner_preview_filter(sem_candidates, owner_id)

    # ✅ If STILL empty → fall back to legacy fuzzy scan (previous behavior, kept intact)
    if not filtered_preview:
        from app.logic.normalize import normalize_prompt as _np  # lazy import safe
        norm2 = _np(normalized_prompt or prompt)
        fallback_keywords = norm2["keywords"]
        fallback = []
        cursor = db.parsed_resumes.find(
            {"ownerUserId": owner_id},
            {
                "_id": 1, "name": 1, "predicted_role": 1, "category": 1, "experience": 1,
                "location": 1, "confidence": 1, "ml_confidence": 1,  # ⬅ added
                "email": 1, "phone": 1, "skills": 1,
                "resume_url": 1, "filename": 1, "currentRole": 1, "resume": 1,
                "semantic_score": 1, "final_score": 1, "raw_text": 1, "projects": 1,
            },
        )
        docs = await cursor.to_list(length=1000)
        for c in docs:
            hay = _candidate_text_blob(c)
            score = 0.0
            for k in (fallback_keywords or []):
                if k in hay: score += 10
            if "developer" in hay or "engineer" in hay: score += 5
            if c.get("semantic_score"): score += float(c["semantic_score"])
            if c.get("final_score"): score += float(c["final_score"])
            if score > 0:
                fallback.append(c)
        filtered_preview = fallback

    # 👇 If user selected filters → strictly apply them (in order)
    if selected_filters:
        filtered_preview = _apply_selected_filters_in_order(
            filtered_preview, selected_filters, normalized_prompt or prompt, keywords or []
        )

    # Summarize matches for UI messaging
    match_meta = _summarize_matches(filtered_preview)
    no_results = match_meta["total"] == 0

    # Step 3: Log analytics (scoped)
    await db.chat_queries.insert_one(
        {
            "prompt": prompt,
            "parsed": parsed,
            "selectedFilters": selected_filters,
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
                "selectedFilters": selected_filters,
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
        "no_results": no_results,
        "no_cvs_uploaded": False,
        "redirect": response.get("redirect"),
        "ui": {
            "primaryMessage": reply_text,
            "query": prompt,
        },
    }
