# 📄 app/routers/history_router.py

from fastapi import APIRouter, Query, HTTPException, Depends
from bson import ObjectId
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from app.utils.mongo import db
from app.routers.auth_router import get_current_user
import re

router = APIRouter()

# --- optional ML matcher (will be used if available) ---
try:
    # Expected to exist per project structure; if not present we fall back to simple filtering
    from app.logic.ml_interface import get_semantic_matches  # type: ignore
except Exception:  # pragma: no cover - safe fallback
    get_semantic_matches = None  # type: ignore


# Utility to convert ObjectId to str
def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc


def _now_display() -> str:
    """
    Human-friendly timestamp similar to the rest of the app.
    Example: 'September 04, 2025 – 03:10 PM'
    """
    return datetime.now().strftime("%B %d, %Y – %I:%M %p")


# -------------------- Lightweight scoring & refiners --------------------

_STOPWORDS = {"and", "or", "the", "a", "an", "in", "with", "of", "for", "to", "on", "at"}

def _tokenize_prompt(prompt: str) -> List[str]:
    # keep words and numbers like "5" and "5+"
    toks = [t.lower() for t in re.findall(r"[a-zA-Z]+|\d+\+?", prompt or "")]
    return [t for t in toks if t and t not in _STOPWORDS]

def _extract_year_bounds(prompt_like: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Very small regex-based extractor used for scoped re-run:
      - '>= 5 years', 'at least 5 years', '5+ years'
      - '<= 8 years', 'up to 8 years', 'at most 8 years'
      - 'between 3 and 6 years'
    Returns (min_years, max_years)
    """
    txt = (prompt_like or "").lower()

    def to_num(s: str) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    # between X and Y
    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+(?:years?|yrs?)", txt)
    if m:
        a = to_num(m.group(1)); b = to_num(m.group(2))
        if a is not None and b is not None:
            return (min(a, b), max(a, b))

    # >= variants
    m = re.search(r"(?:>=|at\s+least|minimum|min)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt)
    min_v = to_num(m.group(1)) if m else None

    # <= variants
    m = re.search(r"(?:<=|at\s+most|no\s+more\s+than|up\s+to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt)
    max_v = to_num(m.group(1)) if m else None

    # N+ years
    m = re.search(r"(\d+)\+\s*(?:years?|yrs?)", txt)
    if m and min_v is None:
        min_v = to_num(m.group(1))

    # strict > / < (without equals)
    if min_v is None:
        m = re.search(r"(?:>\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt)
        if m: min_v = to_num(m.group(1))
    if max_v is None:
        m = re.search(r"(?:<\s*)(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt)
        if m: max_v = to_num(m.group(1))

    return (min_v, max_v)

def _candidate_years(c: Dict[str, Any]) -> float:
    """Try common fields to get years of experience as a float."""
    for k in ("experience", "total_experience_years", "years_of_experience", "experience_years", "yoe"):
        v = c.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.search(r"(\d+(?:\.\d+)?)", v)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    return 0.0

def _compute_simple_score(prompt: str, cand: Dict[str, Any]) -> int:
    """
    Same scoring model as _simple_filter_candidates but for a single candidate,
    so we can enrich scores without reordering.
    """
    tokens = _tokenize_prompt(prompt)
    blob_parts = [
        str(cand.get("name", "")),
        str(cand.get("email", "")),
        str(cand.get("title", "")),
        str(cand.get("role", "")),
        str(cand.get("summary", "")),
        str(cand.get("currentRole", "")),
        str(cand.get("predicted_role", "")),
        str(cand.get("category", "")),
        str(cand.get("location", "")),
        str(cand.get("experience", "")),
    ]
    for key in ("skills", "matchReasons", "highlights", "tags"):
        val = cand.get(key)
        if isinstance(val, list):
            blob_parts.extend([str(x) for x in val])
        elif isinstance(val, str):
            blob_parts.append(val)
    blob = " ".join(blob_parts).lower()

    hits = 0
    for t in tokens:
        if t.endswith("+") and t[:-1].isdigit():
            base = int(t[:-1])
            numbers = [int(n) for n in re.findall(r"\b\d+\b", blob)]
            if any(n >= base for n in numbers):
                hits += 1
        else:
            if t in blob:
                hits += 1
    score = min(100, max(0, 40 + hits * 15))
    return int(score)

def _ensure_scores(prompt: str, candidates: List[Dict[str, Any]]) -> None:
    """
    Guarantee that each candidate carries dynamic score fields expected by UI:
      - final_score
      - prompt_matching_score
    Only fills if missing/falsey, preserves original objects & order.
    """
    for c in candidates or []:
        fs = c.get("final_score")
        pms = c.get("prompt_matching_score")
        if not fs or not pms:
            s = _compute_simple_score(prompt, c)
            if not fs:
                c["final_score"] = s
            if not pms:
                c["prompt_matching_score"] = s
        # Also annotate simple match flags to keep meta counts sane
        if "is_strict_match" not in c:
            c["is_strict_match"] = (c.get("final_score", 0) >= 70)
        if "match_type" not in c:
            c["match_type"] = "exact" if c.get("final_score", 0) >= 85 else ("close" if c.get("final_score", 0) >= 60 else "weak")

def _apply_naive_refiners(prompt: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tiny, non-breaking refiners so re-run behaves closer to the chatbot:
      - experience bounds (>=, <=, between, N+)
      - keyword presence over skills/titles (very light)
    These only make the list stricter; never broaden it.
    """
    if not items:
        return []

    # 1) Experience bounds
    min_y, max_y = _extract_year_bounds(prompt)
    out: List[Dict[str, Any]] = []
    for c in items:
        yrs = _candidate_years(c)
        if min_y is not None and yrs < float(min_y):
            continue
        if max_y is not None and yrs > float(max_y):
            continue
        out.append(c)

    # 2) Light keyword hard filter over skills/title if prompt contains tech-ish tokens
    tokens = [t for t in _tokenize_prompt(prompt) if len(t) >= 3 and not t.isdigit()]
    if not tokens:
        return out

    techish = [t for t in tokens if re.match(r"^[a-z0-9\+\.\#\-]{2,}$", t)]
    if not techish:
        return out

    refined: List[Dict[str, Any]] = []
    for c in out:
        skills = set(str(s).lower() for s in (c.get("skills") or []) if s)
        title = " ".join(
            str(x) for x in [
                c.get("title"), c.get("role"), c.get("predicted_role"),
                (c.get("resume") or {}).get("title"), (c.get("resume") or {}).get("headline")
            ] if x
        ).lower()
        ok = False
        if skills and any(t in skills for t in techish):
            ok = True
        if not ok and title and any(t in title for t in techish):
            ok = True
        if ok:
            refined.append(c)

    # If keyword gate removed everything, fall back to experience-only result (don't be too strict)
    return refined or out


def _simple_filter_candidates(prompt: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback filter if ML matcher is not available.
    Performs a very lightweight keyword match over common candidate fields and
    computes a rough 'final_score' (0..100). This is ONLY a fallback and will
    be superseded automatically when get_semantic_matches is available.
    """
    tokens = _tokenize_prompt(prompt)

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        blob_parts = [
            str(c.get("name", "")),
            str(c.get("email", "")),
            str(c.get("title", "")),
            str(c.get("role", "")),
            str(c.get("summary", "")),
            str(c.get("currentRole", "")),
            str(c.get("predicted_role", "")),
            str(c.get("category", "")),
            str(c.get("location", "")),
            str(c.get("experience", "")),
        ]
        # Flatten common list-like fields
        for key in ("skills", "matchReasons", "highlights", "tags"):
            val = c.get(key)
            if isinstance(val, list):
                blob_parts.extend([str(x) for x in val])
            elif isinstance(val, str):
                blob_parts.append(val)
        blob = " ".join(blob_parts).lower()

        hits = 0
        for t in tokens:
            # support tokens like "5+" or "5"
            if t.endswith("+") and t[:-1].isdigit():
                # treat "5+" as "5" or any number >= 5 present in blob
                base = int(t[:-1])
                numbers = [int(n) for n in re.findall(r"\b\d+\b", blob)]
                if any(n >= base for n in numbers):
                    hits += 1
            else:
                if t in blob:
                    hits += 1

        # simple score: base 40, +15 per hit, cap 100
        score = min(100, max(0, 40 + hits * 15))
        out = dict(c)
        out.setdefault("final_score", score)
        out.setdefault("prompt_matching_score", score)
        out["is_strict_match"] = score >= 70
        out["match_type"] = "exact" if score >= 85 else "close" if score >= 60 else "weak"
        scored.append(out)

    # sort by our computed score desc then name
    scored.sort(key=lambda x: (x.get("final_score", 0), x.get("name", "")), reverse=True)
    # Apply tiny refiners to make re-run stricter when user asks e.g. "Need 5+ years"
    return _apply_naive_refiners(prompt, scored)


# ---------------------- Section-focused helpers (scoped search) ----------------------

def _focus_match(candidate: Dict[str, Any], prompt: str, section: str) -> bool:
    """
    Returns True iff the candidate matches the prompt when searching ONLY within the given section.
    Sections: 'role' | 'experience' | 'education' | 'skills' | 'projects' | 'location' | 'phrases'
    """
    p = (prompt or "").strip().lower()
    if not p:
        return True

    tokens = _tokenize_prompt(p)

    def _in_text(txt: str) -> bool:
        low = txt.lower()
        # token-based contains
        return any(t in low for t in tokens) if tokens else (p in low)

    # role-focused fields
    if section == "role":
        fields = [
            candidate.get("job_role"),
            candidate.get("predicted_role"),
            candidate.get("category"),
            candidate.get("role"),
            candidate.get("title"),
            candidate.get("currentRole"),
        ]
        joined = " ".join([str(x) for x in fields if x])
        return _in_text(joined)

    if section == "skills":
        skills = candidate.get("skills") or []
        skills_text = " ".join([str(s) for s in skills if s])
        return _in_text(skills_text)

    if section == "location":
        loc = str(candidate.get("location") or "")
        return _in_text(loc)

    if section == "education":
        edu_sources: List[str] = []
        # flat fields if present
        for k in ("education", "schools", "degrees", "degree", "university", "college"):
            v = candidate.get(k)
            if isinstance(v, list):
                edu_sources.extend([str(x) for x in v])
            elif isinstance(v, str):
                edu_sources.append(v)
        # from resume subtree
        resume = candidate.get("resume") or {}
        for k in ("education", "schools", "degrees"):
            v = resume.get(k)
            if isinstance(v, list):
                edu_sources.extend([str(x) for x in v])
            elif isinstance(v, str):
                edu_sources.append(v)
        return _in_text(" ".join(edu_sources))

    if section == "projects":
        proj_sources: List[str] = []
        # simple places projects might be found
        for k in ("projects", "project_highlights"):
            v = candidate.get(k)
            if isinstance(v, list):
                proj_sources.extend([str(x) for x in v])
            elif isinstance(v, str):
                proj_sources.append(v)
        resume = candidate.get("resume") or {}
        # workHistory often contains projects/roles
        wh = resume.get("workHistory") or []
        if isinstance(wh, list):
            for item in wh:
                if isinstance(item, dict):
                    for k in ("title", "description", "projects"):
                        v = item.get(k)
                        if isinstance(v, list):
                            proj_sources.extend([str(x) for x in v])
                        elif isinstance(v, str):
                            proj_sources.append(v)
                else:
                    proj_sources.append(str(item))
        return _in_text(" ".join(proj_sources))

    if section == "phrases":
        # exact-ish phrase search across a broad blob
        blob_parts = [
            str(candidate.get("name", "")),
            str(candidate.get("email", "")),
            str(candidate.get("title", "")),
            str(candidate.get("role", "")),
            str(candidate.get("summary", "")),
            str(candidate.get("currentRole", "")),
            str(candidate.get("predicted_role", "")),
            str(candidate.get("category", "")),
            str(candidate.get("location", "")),
            str(candidate.get("experience", "")),
        ]
        for key in ("skills", "matchReasons", "highlights", "tags"):
            val = candidate.get(key)
            if isinstance(val, list):
                blob_parts.extend([str(x) for x in val])
            elif isinstance(val, str):
                blob_parts.append(val)
        resume = candidate.get("resume") or {}
        for k in ("workHistory", "projects", "education", "summary"):
            v = resume.get(k)
            if isinstance(v, list):
                blob_parts.extend([str(x) for x in v])
            elif isinstance(v, str):
                blob_parts.append(v)
        blob = " ".join(blob_parts).lower()
        return p in blob  # phrase presence

    if section == "experience":
        # Only apply experience bounds; if none detected, accept all
        min_y, max_y = _extract_year_bounds(p)
        yrs = _candidate_years(candidate)
        if min_y is not None and yrs < float(min_y):
            return False
        if max_y is not None and yrs > float(max_y):
            return False
        return True

    # Fallback: if the section is unknown, don't filter out
    return True


def _apply_focus_section(matches: List[Dict[str, Any]], prompt: str, focus_section: Optional[str]) -> List[Dict[str, Any]]:
    if not focus_section:
        return matches
    section = focus_section.strip().lower()
    allowed = {"role", "experience", "education", "skills", "projects", "location", "phrases"}
    if section not in allowed:
        return matches
    return [c for c in matches if _focus_match(c, prompt, section)]


# ---------------------- NEW: selectedFilters helpers ----------------------

_ALLOWED_SECTIONS = {"role", "experience", "education", "skills", "projects", "location", "phrases"}

def _normalize_selected_filters(selected_filters: Any) -> List[str]:
    """
    Normalize various selectedFilters payload shapes into a list of allowed section names.
    Supported shapes (mirrors /chatbot/query semantics):
      - list[str]: ["skills","location"]
      - dict with 'sections': {"sections": ["skills","location"]}
      - dict of booleans: {"skills": true, "location": true, "education": false}
    Any unknown values are ignored.
    """
    sections: List[str] = []

    if isinstance(selected_filters, list):
        sections = [str(s).strip().lower() for s in selected_filters]
    elif isinstance(selected_filters, dict):
        if isinstance(selected_filters.get("sections"), list):
            sections = [str(s).strip().lower() for s in selected_filters.get("sections", [])]
        else:
            # treat truthy keys as chosen sections
            for k, v in selected_filters.items():
                if v and isinstance(k, str):
                    sections.append(k.strip().lower())

    # keep only allowed
    return [s for s in sections if s in _ALLOWED_SECTIONS]

def _apply_selected_filters(matches: List[Dict[str, Any]], prompt: str, selected_filters: Any) -> List[Dict[str, Any]]:
    """
    Apply multi-section filtering using the same section-wise matching logic used elsewhere.
    A candidate is kept if it matches in ANY of the selected sections.
    If selected_filters is empty/invalid, returns matches unchanged.
    """
    sections = _normalize_selected_filters(selected_filters)
    if not sections:
        return matches

    kept: List[Dict[str, Any]] = []
    for c in matches:
        if any(_focus_match(c, prompt, sec) for sec in sections):
            kept.append(c)
    return kept


# --------------------------- Routes -----------------------------------

@router.get("/user-history")
async def get_history(
    dateFrom: Optional[str] = Query(None),
    dateTo: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: Optional[str] = Query("latest"),
    current=Depends(get_current_user),
):
    # ✅ Scope to current user
    owner_id = str(getattr(current, "id", None) or getattr(current, "_id", None) or "")

    # ✅ AND all filters together
    conditions: List[Dict[str, Any]] = [{"ownerUserId": owner_id}]

    if search:
        conditions.append({"prompt": {"$regex": search, "$options": "i"}})

    if dateFrom or dateTo:
        time_filter: Dict[str, Any] = {}
        if dateFrom:
            time_filter["$gte"] = datetime.fromisoformat(dateFrom)
        if dateTo:
            time_filter["$lte"] = datetime.fromisoformat(dateTo)
        conditions.append({"timestamp_raw": time_filter})

    query: Dict[str, Any] = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    # ✅ Stable sorting
    if sort == "mostMatches":
        sort_keys = [("totalMatches", -1), ("timestamp_raw", -1), ("_id", -1)]
    elif sort == "oldest":
        sort_keys = [("timestamp_raw", 1), ("_id", 1)]
    else:  # latest (default)
        sort_keys = [("timestamp_raw", -1), ("_id", -1)]

    cursor = db.search_history.find(query).sort(sort_keys)
    results: List[Dict[str, Any]] = []
    async for doc in cursor:
        doc["timestamp"] = doc.get("timestamp_display")
        results.append(serialize_doc(doc))
    return results


@router.get("/history-result/{history_id}")
async def get_history_result(history_id: str, current=Depends(get_current_user)):
    """
    Return a saved history block with candidates.
    ✅ Ensures every candidate carries dynamic match % fields so UI never shows '0% match' blanks.
    ✅ Enforces ownership on fetch.
    """
    try:
        oid = ObjectId(history_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid history id")

    owner_id = str(getattr(current, "id", None) or getattr(current, "_id", None) or "")
    doc = await db.search_history.find_one({"_id": oid, "ownerUserId": owner_id})
    if not doc:
        raise HTTPException(status_code=404, detail="History entry not found")

    # Backfill timestamp for UI
    doc["timestamp"] = doc.get("timestamp_display")

    # ✅ Enrich candidates with dynamic scores if missing (non-destructive, preserves order)
    prompt = str(doc.get("prompt") or "")
    candidates = list(doc.get("candidates") or [])
    _ensure_scores(prompt, candidates)
    doc["candidates"] = candidates

    return serialize_doc(doc)


# ---------------------- NEW: re-run prompt (scoped) ----------------------
@router.post("/rerun/{history_id}")
async def rerun_history_block(history_id: str, payload: Dict[str, Any], current=Depends(get_current_user)):
    """
    Re-run a refined prompt ONLY against the CVs already saved inside this history block.
    The same history document is updated in-place with the narrowed/updated candidate list.
    Response shape is chat-friendly for the popup UI.

    Supports options.focus_section to scope search within a specific section:
      role | experience | education | skills | projects | location | phrases

    Also accepts `selectedFilters` in the payload and applies the same section-wise semantics
    as used by /chatbot/query. When provided, candidates are kept if they match the prompt
    within ANY of the selected sections.
    """
    # Validate history id
    try:
        oid = ObjectId(history_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid history id")

    # Fetch history entry
    h = await db.search_history.find_one({"_id": oid})
    if not h:
        raise HTTPException(status_code=404, detail="History entry not found")

    prompt = (payload or {}).get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    prompt = prompt.strip()

    options = (payload or {}).get("options") or {}
    focus_section = (options.get("focus_section") or "").strip().lower() if isinstance(options, dict) else ""

    # NEW: read selectedFilters (mirrors /chatbot/query semantics)
    selected_filters = (payload or {}).get("selectedFilters", None)

    # Collect whitelist of candidate IDs from the saved block
    original_candidates = h.get("candidates") or []
    id_whitelist: List[str] = []
    for c in original_candidates:
        cid = c.get("_id") or c.get("id")
        if cid is not None:
            try:
                id_whitelist.append(str(cid))
            except Exception:
                pass

    # If we have an ML matcher available, use it with the whitelist.
    # Otherwise, fall back to a simple in-memory keyword filter over the saved candidates.
    matches: List[Dict[str, Any]] = []
    if get_semantic_matches:
        try:
            # The underlying function is expected to accept an id_whitelist to scope search.
            matches = await get_semantic_matches(  # type: ignore
                prompt=prompt,
                id_whitelist=id_whitelist
            )
        except TypeError:
            # If signature differs, try safest call (prompt only) and filter client-side
            tmp = await get_semantic_matches(prompt=prompt)  # type: ignore
            # keep only those inside the saved block, preserving order
            wl = set(id_whitelist)
            for m in tmp or []:
                mid = str(m.get("_id") or m.get("id") or "")
                if mid in wl:
                    matches.append(m)
        except Exception:
            # If ML call fails, degrade gracefully to simple filter
            matches = _simple_filter_candidates(prompt, original_candidates)
        else:
            # ✅ Enrich/normalize ML matches with scores & apply light refiners to mimic chatbot behavior
            _ensure_scores(prompt, matches)
            matches = _apply_naive_refiners(prompt, matches)
    else:
        matches = _simple_filter_candidates(prompt, original_candidates)

    # 🔎 Apply section-focused scope (role/experience/education/skills/projects/location/phrases)
    matches = _apply_focus_section(matches, prompt, focus_section)

    # 🔎 Apply selectedFilters (same semantics as /chatbot/query; OR across chosen sections)
    matches = _apply_selected_filters(matches, prompt, selected_filters)

    # 🔐 NEW: Persist latest match % to candidate profiles (best-effort)
    try:
        for m in matches:
            cid_any = m.get("_id") or m.get("id")
            if cid_any is None:
                continue
            try:
                v = float(m.get("final_score") or m.get("prompt_matching_score") or m.get("score") or 0)
            except Exception:
                v = 0.0
            # Normalize: allow 0..1 inputs (convert to %) then clamp and round
            if 0.0 <= v <= 1.0:
                v *= 100.0
            pct = int(round(max(0.0, min(100.0, v))))
            await db.parsed_resumes.update_one(
                {"_id": ObjectId(cid_any)} if isinstance(cid_any, str) and len(cid_any) in (12, 24) else {"_id": cid_any},
                {"$set": {"prompt_matching_score": pct, "final_score": pct}},
            )
    except Exception:
        # Do not fail the re-run API if persistence has issues
        pass

    # Prepare metadata & timestamps
    ts_raw = datetime.utcnow()
    ts_disp = _now_display()

    match_meta = {
        "strictCount": sum(1 for x in matches if x.get("is_strict_match")),
        "closeCount": sum(1 for x in matches if not x.get("is_strict_match")),
        "total": len(matches),
        "hasExact": any(x.get("match_type") == "exact" for x in matches),
        "hasClose": any(x.get("match_type") == "close" for x in matches),
    }

    # Update the SAME history document in-place
    await db.search_history.update_one(
        {"_id": oid},
        {
            "$set": {
                "prompt": prompt,
                "timestamp_raw": ts_raw,
                "timestamp_display": ts_disp,
                "totalMatches": len(matches),
                "candidates": matches,
                "matchMeta": match_meta,
            }
        },
    )

    # Return a chat-friendly payload for the popup UI
    return {
        "reply": f"Showing {len(matches)} result(s) for your query.",
        "resumes_preview": matches,
        "totalMatches": len(matches),
        "matchMeta": match_meta,
        "no_results": len(matches) == 0,
        "ui": {
            "primaryMessage": f"Updated this block to {len(matches)} candidate(s).",
            "query": prompt,
        },
    }
