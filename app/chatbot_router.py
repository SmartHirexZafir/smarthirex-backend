from fastapi import APIRouter, Request, Depends
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Dict, List, Iterable, Tuple
from bson import ObjectId
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db
from app.routers.auth_router import get_current_user  # ✅ enforce auth
from app.logic.normalize import normalize_prompt, extract_min_years  # ✅ correct imports
import re
import os
import json

router = APIRouter()

# --- Allowed filters (canonical keys) ---------------------------------------
# ✅ Added education mapping and common aliases
_ALLOWED_FILTERS = {
    "role": "role",  # Job Role
    "job_role": "role",
    "jobrole": "role",
    "skills": "skills",  # Skills
    "location": "location",  # Location
    "projects": "projects",  # Projects
    "experience": "experience",  # Experience
    "cv": "cv",  # CV Content Matching
    "cv_content": "cv",
    "cv_content_matching": "cv",
    "education": "education",  # ✅ Education
    "edu": "education",  # mapping for focus_section convenience (handled explicitly as well)
    "phrases": "cv",
}

# Minimal baseline skills for routing-level filtering (kept as final fallback)
_BASELINE_SKILLS_FALLBACK = set(
    [
        "react",
        "aws",
        "node",
        "excel",
        "django",
        "figma",
        "pandas",
        "tensorflow",
        "keras",
        "java",
        "python",
        "pytorch",
        "spark",
        "sql",
        "scikit",
        "mlflow",
        "docker",
        "kubernetes",
        "typescript",
        "next",
        "nextjs",
        "next.js",
        "powerpoint",
        "flora",
        "html",
        "css",
    ]
)

# ✅ Dynamically loaded vocab/synonyms to avoid hardcoding; falls back gracefully
_SKILL_VOCAB: Optional[set] = None
_SYNONYMS: Dict[str, List[str]] = {}


def _resource_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "resources", *parts))


def _load_skill_vocab() -> set:
    global _SKILL_VOCAB
    if _SKILL_VOCAB is not None:
        return _SKILL_VOCAB
    vocab = set()
    try:
        path = _resource_path("skills.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip().lower()
                    if t:
                        vocab.add(t)
    except Exception:
        pass
    # fallback if file missing/empty
    _SKILL_VOCAB = vocab or set(_BASELINE_SKILLS_FALLBACK)
    return _SKILL_VOCAB


def _load_synonyms() -> Dict[str, List[str]]:
    global _SYNONYMS
    if _SYNONYMS:
        return _SYNONYMS
    syn = {}
    try:
        path = _resource_path("synonyms_custom.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # normalize keys/values to lowercase strings
                    for k, v in data.items():
                        key = str(k).strip().lower()
                        if not key:
                            continue
                        if isinstance(v, list):
                            syn[key] = [str(x).strip().lower() for x in v if x]
                        elif isinstance(v, str):
                            syn[key] = [v.strip().lower()]
    except Exception:
        pass

    # ✅ Make sure relational operator synonyms exist for experience parsing
    # (kept here to broaden coverage even if file is missing)
    syn.setdefault(
        "<=",
        [
            "≤",
            "less than or equal to",
            "less than equal to",
            "at most",
            "no more than",
            "up to",
        ],
    )
    syn.setdefault(
        ">=",
        ["≥", "greater than or equal to", "at least", "minimum", "min", "no less than"],
    )
    syn.setdefault("<", ["less than", "under", "below"])
    syn.setdefault(">", ["greater than", "more than", "over", "above"])
    _SYNONYMS = syn
    return _SYNONYMS


# Preload dynamic resources once
_load_skill_vocab()
_load_synonyms()


def _canon_filters(raw_filters: Any) -> List[str]:
    """ Canonicalize selected filter keys and preserve the original order. """
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
    """ Safely convert unknown id shape to ObjectId. Returns None if conversion not possible. """
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
    """ Build a lowercase searchable haystack from candidate fields. """
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
    push(r.get("education"))
    push(r.get("education_details"))
    push(r.get("degrees"))
    push(r.get("universities"))
    push(r.get("institutions"))

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
    """ Try to convert experience into a number of years (handles multiple aliases). """
    for k in (
        "experience",
        "total_experience_years",
        "years_of_experience",
        "experience_years",
        "yoe",
    ):
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


async def _owner_preview_filter(
    items: List[Dict[str, Any]], owner_id: str
) -> List[Dict[str, Any]]:
    """
    Keep only previews that belong to the current owner. Support both string UUID _id and ObjectId _id stored in DB.
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


# ------------------------ keyword helpers / expansion ------------------------
_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}


def _word_to_number(tok: str) -> Optional[int]:
    t = tok.strip().lower()
    if t in _NUM_WORDS:
        return _NUM_WORDS[t]
    # handle hyphenated (e.g., twenty-five)
    if "-" in t:
        parts = t.split("-")
        if all(p in _NUM_WORDS for p in parts):
            return sum(_NUM_WORDS[p] for p in parts)
    return None


def _expand_keywords(keywords: List[str]) -> List[str]:
    """
    Expand keywords using synonyms and number-word normalization.
    Keeps original tokens and adds expansions.
    """
    syn = _load_synonyms()
    out: set = set()
    for k in (keywords or []):
        if not k:
            continue
        k_l = str(k).strip().lower()
        if not k_l:
            continue
        out.add(k_l)
        # number words → digits
        n = _word_to_number(k_l)
        if n is not None:
            out.add(str(n))
        # synonyms file-based
        if k_l in syn:
            for alt in syn[k_l]:
                if alt:
                    out.add(alt.strip().lower())
        # also invert: if k equals a synonym value of some key, include the key
        for root, vals in syn.items():
            if k_l in vals:
                out.add(root)
    return list(out)


# ------------------------ server-side SEQUENTIAL FILTERS ---------------------
def _extract_location_from_prompt(prompt_like: str) -> Optional[str]:
    """ Lightweight location extractor: looks for 'in <place>'. """
    p = " " + (prompt_like or "").lower() + " "
    m = re.search(r"\bin\s+([a-z][a-z .,'-]{1,60})", p)
    if not m:
        return None
    loc = m.group(1).strip()
    # trim trailing punctuation
    loc = re.sub(r"[.,;:!?]+$", "", loc)
    return loc if loc else None


def _looks_like_location_token(tok: str) -> bool:
    """Heuristic to treat a keyword as potential location token."""
    t = tok.strip().lower()
    if not t or len(t) < 3:
        return False
    # avoid obvious operators/units
    if t in {"and", "or", "yrs", "years", "year", "yoe"}:
        return False
    if t in _load_skill_vocab():
        return False
    if t.isdigit():
        return False
    return True


def _extract_year_bounds(prompt_like: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract min/max experience from a prompt using a broader set of phrases:
    - ">= 3 years", "at least 3 years", "minimum 3 years"
    - "<= 5 years", "at most 5 yrs", "up to five years", "no more than 5"
    - "between 3 and 5 years"
    - words for numbers also supported ("four", "ten", "twenty-five")
    Returns (min_years, max_years). Any can be None.
    """
    txt = (prompt_like or "").lower()

    def to_num(m: str) -> Optional[float]:
        m = m.strip().lower()
        if re.match(r"^\d+(\.\d+)?$", m):
            try:
                return float(m)
            except Exception:
                return None
        w = _word_to_number(m)
        return float(w) if w is not None else None

    # between X and Y
    m = re.search(
        r"between\s+([a-z0-9\-\.]+)\s+and\s+([a-z0-9\-\.]+)\s+(?:years?|yrs?)", txt
    )
    if m:
        a = to_num(m.group(1))
        b = to_num(m.group(2))
        if a is not None and b is not None:
            return (min(a, b), max(a, b))

    # <= variants
    syn = _load_synonyms()
    le_phrases = ["<="] + syn.get("<=", [])
    ge_phrases = [">="] + syn.get(">=", [])
    lt_phrases = ["<"] + syn.get("<", [])
    gt_phrases = [">"] + syn.get(">", [])

    # helper to find pattern like "<= 5 years" or "at most five years"
    def find_bound(phrases: List[str], is_max: bool) -> Optional[float]:
        for ph in phrases:
            ph_esc = re.escape(ph)
            pat = rf"{ph_esc}\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)"
            m1 = re.search(pat, txt)
            if m1:
                n = to_num(m1.group(1))
                if n is not None:
                    return n
            # also support "ph ... years" with words in between (e.g., "at most around five years")
            m2 = re.search(rf"{ph_esc}.*?([a-z0-9\-\.]+)\s*(?:years?|yrs?)", txt)
            if m2:
                n = to_num(m2.group(1))
                if n is not None:
                    return n
        return None

    max_v = find_bound(le_phrases, is_max=True)
    min_v = find_bound(ge_phrases, is_max=False)

    # strict < or > 
    if max_v is None:
        max_v = find_bound(lt_phrases, is_max=True)  # for strict "< 5", interpret as <= (inclusive bound) for filtering
    if min_v is None:
        min_v = find_bound(gt_phrases, is_max=False)  # for strict "> 3", interpret as >=

    # fallback to legacy extract_min_years if no min found
    if min_v is None:
        try:
            legacy = extract_min_years(prompt_like)
            if legacy is not None:
                min_v = float(legacy)
        except Exception:
            pass

    return (min_v, max_v)


def _filter_by_role(cands: Iterable[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    wanted_role = (parse_prompt(prompt) or {}).get("job_title")
    return [c for c in cands if _role_matches(wanted_role, c)]


def _filter_by_experience(cands: Iterable[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    """
    Supports >= / <= / between ... and ... with words or digits.
    If only a single value is present (legacy), treats it as >= N.
    """
    min_years, max_years = _extract_year_bounds(prompt)
    out = []
    for c in cands:
        yrs = _candidate_years(c)
        if min_years is not None and yrs < float(min_years):
            continue
        if max_years is not None and yrs > float(max_years):
            continue
        out.append(c)
    return out


def _filter_by_skills(cands: Iterable[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """ Match skills using dynamic vocabulary where possible, falling back to candidate-provided skills. """
    vocab = _load_skill_vocab()
    expanded = set(_expand_keywords(keywords or []))
    # plausible skill tokens: in vocab OR look like tech tokens (letters, "+" allowed)
    focus = [k for k in expanded if (k in vocab or re.match(r"^[a-z0-9\+\.\#\-]{2,}$", k))]
    if not focus:
        return list(cands)
    focus_set = set(focus)
    out: List[Dict[str, Any]] = []
    for c in cands:
        skills = c.get("skills") or []
        skills_lc = set(str(s).strip().lower() for s in skills if s)
        # also scan resume.skill-like fields if provided
        r = c.get("resume") or {}
        extra = r.get("skills") or r.get("technical_skills") or []
        if isinstance(extra, list):
            skills_lc.update(str(s).strip().lower() for s in extra if s)
        elif isinstance(extra, str):
            skills_lc.update(t.strip() for t in extra.lower().split(","))
        if skills_lc & focus_set:
            out.append(c)
    return out


def _filter_by_location(
    cands: Iterable[Dict[str, Any]], prompt: str, keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Match location either via "in <place>" or via location-like tokens in keywords, against multiple location fields on the candidate.
    """
    loc = _extract_location_from_prompt(prompt)
    kw_locs = [k for k in (keywords or []) if _looks_like_location_token(k)]
    if not loc and not kw_locs:
        return list(cands)

    out: List[Dict[str, Any]] = []
    for c in cands:
        r = c.get("resume") or {}
        fields = [
            c.get("location"),
            c.get("city"),
            c.get("country"),
            c.get("address"),
            r.get("location"),
            r.get("city"),
            r.get("country"),
            r.get("address"),
            r.get("work_location"),
            r.get("current_location"),
        ]
        loc_blob = " ".join(str(x) for x in fields if x).lower()
        if not loc_blob:
            continue
        ok = False
        if loc and loc in loc_blob:
            ok = True
        if not ok and kw_locs:
            for t in kw_locs:
                if t.lower() in loc_blob:
                    ok = True
                    break
        if ok:
            out.append(c)
    return out


def _filter_by_projects(cands: Iterable[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """ Project-category-only filter. Looks ONLY at 'projects' field (strings or dicts with name/description). """
    if not keywords:
        return list(cands)
    keys = [k.strip().lower() for k in _expand_keywords([*keywords]) if k and isinstance(k, str)]
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
        # also scan resume.projects if present
        r = c.get("resume") or {}
        rpj = r.get("projects")
        if isinstance(rpj, list):
            for p in rpj:
                if isinstance(p, str):
                    txt += " " + p
                elif isinstance(p, dict):
                    txt += " " + str(p.get("name", "")) + " " + str(p.get("description", ""))
        elif isinstance(rpj, str):
            txt += " " + rpj
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
    keys = [k for k in _expand_keywords(keywords or []) if k and len(k) > 1]
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


def _filter_by_education(cands: Iterable[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """ Education-only filter: match degree names, institutions, majors against keywords. """
    if not keywords:
        return list(cands)
    keys = [k.strip().lower() for k in _expand_keywords([*keywords]) if k and isinstance(k, str)]
    # common degree tokens to widen coverage
    degree_tokens = {
        "bachelor",
        "bachelors",
        "ba",
        "bs",
        "bsc",
        "be",
        "b.e",
        "b.tech",
        "btech",
        "master",
        "masters",
        "ms",
        "msc",
        "m.tech",
        "mtech",
        "mba",
        "mphil",
        "phd",
        "doctorate",
        "associate",
        "diploma",
    }
    keys_set = set(keys) | degree_tokens

    def edu_blob(c: Dict[str, Any]) -> str:
        r = c.get("resume") or {}
        parts: List[str] = []
        for fld in ("education", "education_details", "degrees", "universities", "institutions"):
            v = r.get(fld) or c.get(fld)
            if not v:
                continue
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        parts.append(str(x.get("degree", "")))
                        parts.append(str(x.get("field", "")))
                        parts.append(str(x.get("institution", "")))
                        parts.append(str(x.get("university", "")))
                    else:
                        parts.append(str(x))
            elif isinstance(v, dict):
                parts.extend(str(x) for x in v.values())
            else:
                parts.append(str(v))
        return " ".join(parts).lower()

    out: List[Dict[str, Any]] = []
    for c in cands:
        blob = edu_blob(c)
        if blob and any(k in blob for k in keys_set):
            out.append(c)
    return out


def _apply_selected_filters_in_order(
    items: List[Dict[str, Any]],
    selected: List[str],
    prompt_like: str,
    keywords: List[str],
) -> List[Dict[str, Any]]:
    """
    ✅ CATEGORY-FIRST FILTERING: Apply filters in strict priority order with early termination.
    Priority: role → experience → skills → location → education → projects → phrases
    
    If role filter exists and fails → candidate is excluded immediately (no other checks).
    This dramatically speeds up filtering by avoiding unnecessary field scans.
    """
    working = list(items)
    if not working or not selected:
        return working

    # ✅ CRITICAL: Apply filters in priority order with early termination
    # 1) ROLE (highest priority - if fails, skip candidate entirely)
    if "role" in selected:
        working = _filter_by_role(working, prompt_like)
        if not working:
            return []  # Early termination: no candidates match role

    # 2) EXPERIENCE (only applied if role passed or role not selected)
    if "experience" in selected:
        working = _filter_by_experience(working, prompt_like)
        if not working:
            return []  # Early termination

    # 3) SKILLS (only applied if role+experience passed)
    if "skills" in selected:
        working = _filter_by_skills(working, keywords)
        if not working:
            return []  # Early termination

    # 4) LOCATION (only applied if previous filters passed)
    if "location" in selected:
        working = _filter_by_location(working, prompt_like, keywords)
        if not working:
            return []  # Early termination

    # 5) EDUCATION (only applied if previous filters passed)
    if "education" in selected:
        working = _filter_by_education(working, keywords)
        if not working:
            return []  # Early termination

    # 6) PROJECTS (only applied if previous filters passed)
    if "projects" in selected:
        working = _filter_by_projects(working, keywords)
        if not working:
            return []  # Early termination

    # 7) PHRASES/CV (only applied if previous filters passed)
    if "cv" in selected or "phrases" in selected:
        working = _filter_by_cv_content(working, prompt_like, keywords)
        if not working:
            return []  # Early termination

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


# ------------------------ scoring/metrics for cards --------------------------
def _to_percent(val: Optional[float]) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
    except Exception:
        return 0.0
    if 0.0 <= v <= 1.0:
        return round(v * 100.0, 1)
    # clamp to [0,100]
    v = max(0.0, min(v, 100.0))
    return round(v, 1)


def _compute_prompt_match_score(c: Dict[str, Any], prompt_like: str, keywords: List[str]) -> float:
    """
    Deterministic prompt-match scoring based on keyword hits + optional semantic_score.
    Scaled to 0..100 for UI.
    """
    blob = _candidate_text_blob(c)
    if not blob:
        return 0.0

    p = (prompt_like or "").lower().strip()
    keys = [k for k in _expand_keywords(keywords or []) if k and len(k) > 1]
    hits = 0.0

    for k in keys:
        if k in blob:
            hits += 1.0

    # small bonus for prompt substring
    if p and p in blob:
        hits += 1.5

    # include semantic_score gently if present (assumed 0..1 or 0..100)
    sem = c.get("semantic_score")
    if sem is not None:
        hits += (_to_percent(sem) / 100.0) * 1.5  # max +1.5

    # Normalize: assume up to ~10 meaningful tokens contributes strongly
    score = min(100.0, round((hits / 10.0) * 100.0, 1))
    return score


def _annotate_scores(preview: List[Dict[str, Any]], prompt_like: str, keywords: List[str]) -> None:
    """
    Ensure every candidate has a dynamic score the UI understands.
    Frontend falls back in this order: final_score → prompt_matching_score → score → match_score.
    To avoid '0% match' blanks, we set both synonyms if they're missing.
    """
    for c in preview or []:
        # ✅ Fix: Compute prompt matching score - prefer existing semantic_score if available, otherwise compute
        # If candidate already has a semantic_score from ML matching, use it as base
        existing_semantic = c.get("semantic_score")
        existing_final = c.get("final_score")
        existing_prompt = c.get("prompt_matching_score")
        
        # If we have a semantic score from ML matching, prefer it (it's more accurate)
        if existing_semantic is not None and isinstance(existing_semantic, (int, float)):
            # Normalize semantic_score to 0-100 if needed
            if existing_semantic <= 1.0:
                semantic_pct = existing_semantic * 100.0
            else:
                semantic_pct = float(existing_semantic)
            semantic_pct = max(0.0, min(100.0, semantic_pct))
            
            # Use semantic score if no existing final_score, or if semantic is higher
            if not existing_final or semantic_pct > existing_final:
                computed = int(round(semantic_pct))
            else:
                computed = int(round(_compute_prompt_match_score(c, prompt_like, keywords)))
        else:
            # Compute using keyword matching
            computed = int(round(_compute_prompt_match_score(c, prompt_like, keywords)))
        
        # Prefer existing values; only fill if missing/falsey
        if not c.get("final_score") or c.get("final_score") == 0:
            c["final_score"] = computed
        if not c.get("prompt_matching_score") or c.get("prompt_matching_score") == 0:
            c["prompt_matching_score"] = computed
        # Keep the lightweight debug field too (harmless if unused)
        c["prompt_match_score"] = computed

        # ✅ Fix: Role Prediction Confidence - check multiple fields with proper priority
        role_conf = None
        # Priority: role_confidence (0-1) > ml_confidence (0-1) > confidence (0-1 or 0-100) > role_confidence_pct/100
        if isinstance(c.get("role_confidence"), (int, float)) and float(c.get("role_confidence", 0)) > 0:
            role_conf = float(c.get("role_confidence"))
        elif isinstance(c.get("ml_confidence"), (int, float)) and float(c.get("ml_confidence", 0)) > 0:
            conf_val = float(c.get("ml_confidence"))
            role_conf = conf_val / 100.0 if conf_val > 1.0 else conf_val
        elif isinstance(c.get("confidence"), (int, float)) and float(c.get("confidence", 0)) > 0:
            conf_val = float(c.get("confidence"))
            role_conf = conf_val / 100.0 if conf_val > 1.0 else conf_val
        elif isinstance(c.get("role_confidence_pct"), (int, float)) and float(c.get("role_confidence_pct", 0)) > 0:
            role_conf = float(c.get("role_confidence_pct")) / 100.0
        # Fallback: if we have a predicted_role but no confidence, assign reasonable default
        elif c.get("predicted_role") or c.get("ml_predicted_role") or c.get("category"):
            role_conf = 0.7  # 70% default confidence if role exists but confidence missing
        
        c["role_prediction_confidence"] = _to_percent(role_conf) if role_conf is not None else 0.0


def _ensure_related_roles(preview: List[Dict[str, Any]]) -> None:
    """
    Ensure each candidate has a 'related_roles' array.
    - If missing/empty, compute related roles inline using semantic similarity.
    - Normalize scores to percent (0..100) and keys to {'role','match'}.
    - Provide 3–5 items from server side; client may clamp to 3.
    """
    # Lazy import to avoid cyclic deps
    try:
        from app.logic.ml_interface import get_full_roles_for_suggestions, embedding_model, util
        has_ml = True
    except Exception:
        has_ml = False

    for c in preview or []:
        rr = c.get("related_roles") or c.get("relatedRoles")
        needs_infer = not isinstance(rr, list) or len(rr) == 0
        
        # ✅ Fix: Compute related roles inline if missing
        if needs_infer:
            predicted_role = c.get("predicted_role") or c.get("ml_predicted_role") or c.get("category")
            if predicted_role and has_ml:
                try:
                    # Compute related roles using semantic similarity
                    pred_role_lc = str(predicted_role).strip().lower()
                    if pred_role_lc and pred_role_lc not in {"n/a", "na", "none", "-", "unknown", "not specified"}:
                        pred_embed = embedding_model.encode(predicted_role)
                        scores = []
                        full_roles = get_full_roles_for_suggestions()
                        for role in full_roles:
                            role_clean = str(role).strip().lower()
                            if role_clean == pred_role_lc:
                                continue
                            role_embed = embedding_model.encode(role)
                            rscore = util.cos_sim(pred_embed, role_embed)[0][0].item()
                            # Only include roles with similarity > 0.3
                            if rscore > 0.3:
                                scores.append({"role": role, "match": round(rscore * 100, 2)})
                        scores.sort(key=lambda x: x["match"], reverse=True)
                        rr = scores[:5]  # Top 5 related roles
                    else:
                        rr = []
                except Exception:
                    rr = []
            else:
                rr = []

        # Normalize to list of dicts with 'role' and percent 'match'
        norm_list: List[Dict[str, Any]] = []
        if isinstance(rr, list):
            for it in rr:
                if isinstance(it, dict):
                    role = it.get("role") or it.get("title") or it.get("name")
                    score = it.get("match")
                    if score is None:
                        score = it.get("score")
                else:
                    role = str(it)
                    score = None
                if role:  # Only add if role exists
                    pct = _to_percent(score) if isinstance(score, (int, float)) else 0.0
                    norm_list.append({"role": role, "match": pct})

        # attach normalized (ensure at least empty list)
        c["related_roles"] = norm_list if norm_list else []
        c["relatedRoles"] = [{"role": x["role"], "score": round(x["match"]/100.0, 4)} for x in norm_list]


# ------------------------ per-candidate match flag helpers -------------------
# (Used ONLY to compute is_strict_match & match_type as requested)
def _candidate_matches_role_single(c: Dict[str, Any], prompt_like: str) -> bool:
    wanted_role = (parse_prompt(prompt_like) or {}).get("job_title")
    return _role_matches(wanted_role, c)


def _candidate_matches_experience_single(c: Dict[str, Any], prompt_like: str) -> bool:
    min_years, max_years = _extract_year_bounds(prompt_like)
    yrs = _candidate_years(c)
    if min_years is not None and yrs < float(min_years):
        return False
    if max_years is not None and yrs > float(max_years):
        return False
    return True


def _candidate_matches_skills_single(c: Dict[str, Any], keywords: List[str]) -> bool:
    vocab = _load_skill_vocab()
    expanded = set(_expand_keywords(keywords or []))
    focus = [k for k in expanded if (k in vocab or re.match(r"^[a-z0-9\+\.\#\-]{2,}$", k))]
    if not focus:
        # if no concrete skill terms, treat as not constraining
        return True
    focus_set = set(focus)
    skills = c.get("skills") or []
    skills_lc = set(str(s).strip().lower() for s in skills if s)
    r = c.get("resume") or {}
    extra = r.get("skills") or r.get("technical_skills") or []
    if isinstance(extra, list):
        skills_lc.update(str(s).strip().lower() for s in extra if s)
    elif isinstance(extra, str):
        skills_lc.update(t.strip() for t in extra.lower().split(","))
    return bool(skills_lc & focus_set)


def _candidate_matches_location_single(c: Dict[str, Any], prompt_like: str, keywords: List[str]) -> bool:
    loc = _extract_location_from_prompt(prompt_like)
    kw_locs = [k for k in (keywords or []) if _looks_like_location_token(k)]
    if not loc and not kw_locs:
        return True
    r = c.get("resume") or {}
    fields = [
        c.get("location"),
        c.get("city"),
        c.get("country"),
        c.get("address"),
        r.get("location"),
        r.get("city"),
        r.get("country"),
        r.get("address"),
        r.get("work_location"),
        r.get("current_location"),
    ]
    loc_blob = " ".join(str(x) for x in fields if x).lower()
    if not loc_blob:
        return False
    if loc and loc in loc_blob:
        return True
    for t in kw_locs:
        if t.lower() in loc_blob:
            return True
    return False


def _candidate_matches_projects_single(c: Dict[str, Any], keywords: List[str]) -> bool:
    if not keywords:
        return True
    keys = [k.strip().lower() for k in _expand_keywords([*keywords]) if k and isinstance(k, str)]
    txt = ""
    projs = c.get("projects", [])
    if isinstance(projs, list):
        for p in projs:
            if isinstance(p, str):
                txt += " " + p
            elif isinstance(p, dict):
                txt += " " + str(p.get("name", "")) + " " + str(p.get("description", ""))
    r = c.get("resume") or {}
    rpj = r.get("projects")
    if isinstance(rpj, list):
        for p in rpj:
            if isinstance(p, str):
                txt += " " + p
            elif isinstance(p, dict):
                txt += " " + str(p.get("name", "")) + " " + str(p.get("description", ""))
    elif isinstance(rpj, str):
        txt += " " + rpj
    txt = txt.lower()
    return bool(txt and any(k in txt for k in keys))


def _candidate_matches_cv_single(c: Dict[str, Any], prompt_like: str, keywords: List[str]) -> bool:
    p = (prompt_like or "").lower().strip()
    keys = [k for k in _expand_keywords(keywords or []) if k and len(k) > 1]
    blob = (
        str(c.get("raw_text") or "")
        + " "
        + str((c.get("resume") or {}).get("summary") or "")
        + " "
        + str((c.get("resume") or {}).get("workHistory") or "")
    ).lower()
    if not blob:
        return False
    if p and p in blob:
        return True
    if keys and any(k in blob for k in keys):
        return True
    return False


def _candidate_matches_education_single(c: Dict[str, Any], keywords: List[str]) -> bool:
    if not keywords:
        # no edu tokens provided → not constraining
        return True
    keys = [k.strip().lower() for k in _expand_keywords([*keywords]) if k and isinstance(k, str)]
    degree_tokens = {
        "bachelor",
        "bachelors",
        "ba",
        "bs",
        "bsc",
        "be",
        "b.e",
        "b.tech",
        "btech",
        "master",
        "masters",
        "ms",
        "msc",
        "m.tech",
        "mtech",
        "mba",
        "mphil",
        "phd",
        "doctorate",
        "associate",
        "diploma",
    }
    keys_set = set(keys) | degree_tokens
    r = c.get("resume") or {}
    parts: List[str] = []
    for fld in ("education", "education_details", "degrees", "universities", "institutions"):
        v = r.get(fld) or c.get(fld)
        if not v:
            continue
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    parts.append(str(x.get("degree", "")))
                    parts.append(str(x.get("field", "")))
                    parts.append(str(x.get("institution", "")))
                    parts.append(str(x.get("university", "")))
                else:
                    parts.append(str(x))
        elif isinstance(v, dict):
            parts.extend(str(x) for x in v.values())
        else:
            parts.append(str(v))
    blob = " ".join(parts).lower()
    return bool(blob and any(k in blob for k in keys_set))


def _compute_match_flags_for_candidate(
    c: Dict[str, Any],
    selected_filters: List[str],
    prompt_like: str,
    expanded_keywords: List[str],
) -> Tuple[bool, str]:
    """
    Compute (is_strict_match, match_type).
    - If selected_filters provided: EXACT when candidate satisfies *all* active namespaces.
    - If no selected_filters: EXACT when candidate has clear prompt/keyword hits; otherwise CLOSE.
    """
    # When filters are selected, require all of them to be satisfied
    if selected_filters:
      ok_all = True
      for f in selected_filters:
          if f == "role":
              ok_all = ok_all and _candidate_matches_role_single(c, prompt_like)
          elif f == "experience":
              ok_all = ok_all and _candidate_matches_experience_single(c, prompt_like)
          elif f == "skills":
              ok_all = ok_all and _candidate_matches_skills_single(c, expanded_keywords)
          elif f == "location":
              ok_all = ok_all and _candidate_matches_location_single(c, prompt_like, expanded_keywords)
          elif f == "projects":
              ok_all = ok_all and _candidate_matches_projects_single(c, expanded_keywords)
          elif f == "cv":
              ok_all = ok_all and _candidate_matches_cv_single(c, prompt_like, expanded_keywords)
          elif f == "education":
              ok_all = ok_all and _candidate_matches_education_single(c, expanded_keywords)
          if not ok_all:
              break
      return (ok_all, "exact" if ok_all else "close")

    # No explicit filters → rely on prompt/keyword presence for EXACT
    blob = _candidate_text_blob(c)
    p = (prompt_like or "").lower().strip()
    keys = [k for k in (expanded_keywords or []) if k and len(k) > 1]
    hit = False
    if blob:
        if p and p in blob:
            hit = True
        elif any(k in blob for k in keys):
            hit = True
    return (hit, "exact" if hit else "close")


# ------------------------ structured options derivation ----------------------
def _derive_structured_options(
    prompt_like: str,
    expanded_keywords: List[str],
    selected_filters: List[str],
) -> Dict[str, Any]:
    """
    Build a structured 'options' payload from the prompt and keywords so downstream components (including ML)
    can optionally use strong hints without hardcoding.
    """
    vocab = _load_skill_vocab()
    min_years, max_years = _extract_year_bounds(prompt_like)
    loc = _extract_location_from_prompt(prompt_like)

    # Extract likely skills from expanded keywords
    skills_hints = sorted({k for k in expanded_keywords if k in vocab})

    # Education tokens (lightweight, same set used in filter)
    degree_tokens = {
        "bachelor",
        "bachelors",
        "ba",
        "bs",
        "bsc",
        "be",
        "b.e",
        "b.tech",
        "btech",
        "master",
        "masters",
        "ms",
        "msc",
        "m.tech",
        "mtech",
        "mba",
        "mphil",
        "phd",
        "doctorate",
        "associate",
        "diploma",
    }
    edu_terms = sorted({k for k in expanded_keywords if k in degree_tokens})

    return {
        "location": loc,
        "min_years": min_years,
        "max_years": max_years,
        "skills": skills_hints,
        "projects_required": "projects" in (selected_filters or []),
        "education_terms": edu_terms,
        "cv_keywords": expanded_keywords,
        "filters_order": selected_filters or [],
    }


@router.post("/query")
async def handle_chatbot_query(
    request: Request,
    current_user=Depends(get_current_user),
):
    """
    Receives a natural language prompt from user, identifies intent, processes it, and responds accordingly.
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
            "ui": {
                "primaryMessage": "Invalid request payload.",
                "query": "",
                "showCandidates": False,  # ✅ ensure UI hides sections
                "showRouter": False,
                "toast": {"message": "Invalid request payload.", "type": "error", "durationMs": 3500},  # ✅ auto-dismiss hint
            },
        }

    prompt = (data.get("prompt") or "").strip()

    selected_filters_raw = (
        data.get("selected_filters") or data.get("filters") or data.get("selectedFilters") or []
    )
    selected_filters = _canon_filters(selected_filters_raw)

    # ✅ NEW: accept options.focus_section and enforce single-section filtering
    req_options = data.get("options") or {}
    focus_section_raw = str(req_options.get("focus_section", "") or "").strip().lower()
    if focus_section_raw:
        mapped = _ALLOWED_FILTERS.get(focus_section_raw, "")
        # special-case synonyms: 'phrases' should map to cv
        if not mapped and focus_section_raw == "phrases":
            mapped = "cv"
        if mapped:
            selected_filters = [mapped]  # 🔒 enforce ONLY this section

    if not prompt:
        return {
            "reply": "Prompt is empty.",
            "resumes_preview": [],
            "no_results": True,
            "no_cvs_uploaded": False,
            "matchMeta": {"strictCount": 0, "closeCount": 0, "total": 0, "hasExact": False, "hasClose": False},
            "ui": {
                "primaryMessage": "Prompt is empty.",
                "query": "",
                "showCandidates": False,  # ✅ hide candidates/router on empty prompt
                "showRouter": False,
                "toast": {"message": "Please enter a prompt.", "type": "info", "durationMs": 3500},
            },
        }

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
            "ui": {
                "primaryMessage": "You haven't uploaded any resumes yet.",
                "query": prompt,
                "showCandidates": False,  # ✅ do not render candidate list/router
                "showRouter": False,
                "toast": {"message": "No CVs uploaded.", "type": "warning", "durationMs": 3500},
            },
        }

    # --- normalization (use client-provided if available, else compute)
    normalized_prompt = (data.get("normalized_prompt") or "").strip()
    keywords = data.get("keywords")

    if not normalized_prompt or not isinstance(keywords, list):
        norm = normalize_prompt(prompt)  # 🔧 correct function
        normalized_prompt = norm["normalized_prompt"]
        keywords = norm["keywords"]

    # ✅ Broaden keywords for filtering coverage (numbers-in-words, synonyms)
    expanded_keywords = _expand_keywords(keywords or [])

    # Step 1: Parse intent (enrich parsed with normalized info for downstream)
    parsed = parse_prompt(prompt) or {}
    parsed.setdefault("normalized_prompt", normalized_prompt)
    parsed.setdefault("keywords", keywords)
    parsed.setdefault("ownerUserId", owner_id)  # 👈 pass owner for downstream filtering
    # 👇 include selected filters (order preserved)
    parsed["selectedFilters"] = selected_filters

    # ✅ NEW: derive structured 'options' for downstream (non-breaking)
    options = _derive_structured_options(normalized_prompt or prompt, expanded_keywords, selected_filters)
    # Include a role hint if parser detected one
    if "job_title" in parsed and parsed["job_title"]:
        options["role"] = parsed["job_title"]
    # ✅ reflect requested focus section in options for downstream components
    if focus_section_raw:
        options["focus_section"] = _ALLOWED_FILTERS.get(
            focus_section_raw, "cv" if focus_section_raw == "phrases" else focus_section_raw
        )
    parsed["options"] = options  # build_response may consume or ignore

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
            "keywords_expanded": expanded_keywords,  # ✅ visibility for UI/debug
            "matchMeta": {"strictCount": 0, "closeCount": 0, "total": 0, "hasExact": False, "hasClose": False},
            "no_results": True,
            "no_cvs_uploaded": False,
            "ui": {
                "primaryMessage": "Sorry, something went wrong while processing your query.",
                "query": prompt,
                "showCandidates": False,  # ✅ avoid loading sections on error
                "showRouter": False,
                "toast": {"message": "Processing error.", "type": "error", "durationMs": 3500},
            },
        }

    # ✅ Safety post-filter: keep only resumes owned by current user
    raw_preview = response.get("resumes_preview", []) or []
    filtered_preview = await _owner_preview_filter(raw_preview, owner_id)

    # ✅ NEW: Semantic fallback (owner-scoped) so cards get semantic_score, related_roles, role etc.
    # If still empty after the first pass, use get_semantic_matches instead of the old fuzzy DB scan.
    if not filtered_preview:
        try:
            from app.logic.ml_interface import get_semantic_matches  # lazy import to avoid cycles

            # Try to pass options/filters if supported; fall back gracefully
            try:
                sem_candidates = await get_semantic_matches(
                    normalized_prompt or prompt,
                    owner_user_id=owner_id,
                    normalized_prompt=normalized_prompt,
                    keywords=expanded_keywords,
                    options=options,  # may be ignored if signature doesn't accept
                    selected_filters=selected_filters,
                )
            except TypeError:
                # Older signature without options/selected_filters
                sem_candidates = await get_semantic_matches(
                    normalized_prompt or prompt,
                    owner_user_id=owner_id,
                    normalized_prompt=normalized_prompt,
                    keywords=expanded_keywords,
                )
        except Exception:
            sem_candidates = []  # keep ownership safety

        filtered_preview = await _owner_preview_filter(sem_candidates, owner_id)

    # ✅ If STILL empty → fall back to legacy fuzzy scan (previous behavior, kept intact)
    if not filtered_preview:
        from app.logic.normalize import normalize_prompt as _np  # lazy import safe

        norm2 = _np(normalized_prompt or prompt)
        fallback_keywords = _expand_keywords(norm2["keywords"])
        fallback = []
        cursor = db.parsed_resumes.find(
            {
                "ownerUserId": owner_id,
            },
            {
                "_id": 1,
                "name": 1,
                "predicted_role": 1,
                "category": 1,
                "experience": 1,
                "location": 1,
                "confidence": 1,
                "ml_confidence": 1,  # ⬅ added
                "email": 1,
                "phone": 1,
                "skills": 1,
                "resume_url": 1,
                "filename": 1,
                "currentRole": 1,
                "resume": 1,
                "semantic_score": 1,
                "final_score": 1,
                "raw_text": 1,
                "projects": 1,
                "education": 1,
            },
        )
        docs = await cursor.to_list(length=1000)
        for c in docs:
            hay = _candidate_text_blob(c)
            score = 0.0
            for k in (fallback_keywords or []):
                if k in hay:
                    score += 10
            if "developer" in hay or "engineer" in hay:
                score += 5
            if c.get("semantic_score"):
                score += float(c["semantic_score"])
            if c.get("final_score"):
                score += float(c["final_score"])
            if score > 0:
                fallback.append(c)
        filtered_preview = fallback

    # 👇 If user selected filters → strictly apply them (in order)
    if selected_filters:
        filtered_preview = _apply_selected_filters_in_order(
            filtered_preview, selected_filters, normalized_prompt or prompt, expanded_keywords or []
        )

    # ✅ NEW: Emit consistent match flags per candidate (exact vs close)
    # Rule: exact when ALL active namespaces satisfied; close otherwise.
    # If no selected filters, treat prompt/keyword hit as exact; else close.
    for cand in filtered_preview:
        strict, mtype = _compute_match_flags_for_candidate(
            cand,
            selected_filters=selected_filters,
            prompt_like=normalized_prompt or prompt,
            expanded_keywords=expanded_keywords or [],
        )
        cand["is_strict_match"] = bool(strict)
        cand["match_type"] = mtype  # "exact" | "close"

    # ✅ Annotate card metrics: ensure dynamic % present with UI-recognized keys
    _annotate_scores(filtered_preview, normalized_prompt or prompt, expanded_keywords)

    # ✅ Ensure/normalize related roles (infer when missing)
    _ensure_related_roles(filtered_preview)

    # ✅ NEW (per requirement 4): Persist latest match % back to candidate documents
    try:
        for c in filtered_preview or []:
            cid = str(c.get("_id") or c.get("id") or "")
            if not cid:
                continue
            raw = c.get("final_score")
            if raw is None:
                raw = c.get("prompt_matching_score")
            if raw is None:
                continue
            try:
                pct = float(raw)
            except Exception:
                continue
            pct = max(0.0, min(100.0, pct))
            pct_int = int(round(pct))
            await db.parsed_resumes.update_one(
                {"_id": cid, "ownerUserId": owner_id},
                {"$set": {"prompt_matching_score": pct_int, "final_score": pct_int}},
            )
    except Exception:
        # best-effort only; never block chat on persistence
        pass

    # Summarize matches for UI messaging
    match_meta = _summarize_matches(filtered_preview)
    no_results = match_meta["total"] == 0

    # Step 3: Log analytics (scoped)
    await db.chat_queries.insert_one(
        {
            "prompt": prompt,
            "parsed": parsed,
            "selectedFilters": selected_filters,
            "options": options,
            "response_preview_count": len(filtered_preview),
            "timestamp": datetime.now(timezone.utc),
            "ownerUserId": owner_id,
            "matchMeta": match_meta,
        }
    )

    # Step 4: Save history for filter intents (prevent duplicates)
    if (parsed.get("intent") == "filter_cv") and filtered_preview:
        # ✅ Prevent duplicate saves: Check if a history entry with the same prompt
        # was saved within the last 5 seconds by the same user
        recent_cutoff = datetime.now(timezone.utc) - timedelta(seconds=5)
        existing = await db.search_history.find_one({
            "ownerUserId": owner_id,
            "prompt": prompt,
            "timestamp_raw": {"$gte": recent_cutoff},
        })
        
        # Only save if no recent duplicate exists
        if not existing:
            timestamp_raw = datetime.now(timezone.utc)
            timestamp_display = datetime.now().strftime("%B  %d, %Y – %I:%M %p")
            await db.search_history.insert_one(
                {
                    "prompt": prompt,
                    "parsed": parsed,
                    "selectedFilters": selected_filters,
                    "options": options,
                    "timestamp_raw": timestamp_raw,
                    "timestamp_display": timestamp_display,
                    "totalMatches": len(filtered_preview),
                    "candidates": filtered_preview,  # ⬅ carries final_score & prompt_matching_score now
                    "ownerUserId": owner_id,
                    "matchMeta": match_meta,
                }
            )

    # Step 5: Context-aware reply text for UI banners
    count = match_meta["total"]
    reply_text = f"Showing {count} result{'s' if count != 1 else ''} for your query."
    if count == 0:
        reply_text = "Showing 0 results for your query."

    # ✅ Add UI control flags so frontend can hide router / candidate sections when 0 results
    ui_payload = {
        "primaryMessage": reply_text,
        "query": prompt,
        "showCandidates": not no_results,  # hide when no results
        "showRouter": not no_results,  # hide when no results
        # ✅ optional toast hint with auto-dismiss duration 3–4s; frontend can ignore or use
        "toast": {"message": reply_text, "type": "info", "durationMs": 3500},
    }

    return {
        "reply": reply_text,  # for the chat bubble
        "resumes_preview": filtered_preview,  # cards
        "normalized_prompt": normalized_prompt,  # extra context for UI
        "keywords": keywords,  # original keywords
        "keywords_expanded": expanded_keywords,  # ✅ broadened keywords (debug/UX)
        "selectedFilters": selected_filters,  # ✅ echo back selection
        "options": options,  # ✅ structured hints for UI/debug
        "matchMeta": match_meta,  # counts for banners
        "no_results": no_results,
        "no_cvs_uploaded": False,
        "redirect": response.get("redirect"),
        "ui": ui_payload,
    }
