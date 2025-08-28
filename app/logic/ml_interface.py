import joblib
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from app.utils.mongo import db
from sentence_transformers import SentenceTransformer, util
from app.logic.normalize import normalize_prompt


# -----------------------------
# Load classic ML assets (LAZY)
# -----------------------------
MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

# lazy singletons (boot-safe)
_model: Optional[BaseEstimator] = None
_vectorizer: Optional[TfidfVectorizer] = None
_label_encoder: Optional[LabelEncoder] = None
_load_error: Optional[Exception] = None

def get_classic_assets() -> Tuple[Optional[BaseEstimator], Optional[TfidfVectorizer], Optional[LabelEncoder]]:
    """
    Lazy-initialize classic ML artifacts. If files are missing/corrupt,
    degrade gracefully by returning (None, None, None).
    """
    global _model, _vectorizer, _label_encoder, _load_error
    if _model is not None or _vectorizer is not None or _label_encoder is not None:
        return _model, _vectorizer, _label_encoder
    try:
        from joblib import load as joblib_load
        _model = joblib_load(MODEL_PATH)
        _vectorizer = joblib_load(VECTORIZER_PATH)
        _label_encoder = joblib_load(ENCODER_PATH)
    except Exception as e:
        _load_error = e
        _model = _vectorizer = _label_encoder = None
    return _model, _vectorizer, _label_encoder


# -------------------------------------
# Robust semantic model loader (offline)
# -------------------------------------
MODEL_ID = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_MODEL_DIR = os.getenv(
    "LOCAL_SENTENCE_MODEL_DIR",
    os.path.join("app", "ml_models", "all-MiniLM-L6-v2")
)

def _load_embedding_model() -> SentenceTransformer:
    # 1) Try online (default)
    try:
        return SentenceTransformer(MODEL_ID)
    except Exception:
        pass
    # 2) Try local directory
    try:
        return SentenceTransformer(LOCAL_MODEL_DIR, local_files_only=True)
    except Exception:
        pass
    # 3) Try cache-only
    return SentenceTransformer(MODEL_ID, local_files_only=True)

embedding_model = _load_embedding_model()

# ✅ Extended real-world roles (expanded with design/front-end families)
extended_roles = [
    # AI/Data/ML
    "AI Engineer", "ML Engineer", "NLP Engineer", "Deep Learning Engineer",
    "MLOps Engineer", "Data Analyst", "Data Engineer", "Computer Vision Engineer",
    "Research Scientist", "Machine Learning Engineer", "Data Scientist",
    "Business Analyst", "Software Engineer", "Full Stack Engineer",
    # Frontend / Web / Design
    "Front-End Developer", "Frontend Developer", "Web Developer", "Web Designer",
    "UI Designer", "UX Designer", "UI/UX Designer", "Product Designer",
    "Graphic Designer", "Visual Designer"
]

def get_role_vocab() -> List[str]:
    """
    Build role vocabulary at runtime:
    - Always include extended_roles
    - If LabelEncoder is available, extend with its classes
    Lowercased & deduped for matching.
    """
    roles: List[str] = list(dict.fromkeys(extended_roles))
    _, _, le = get_classic_assets()
    if le is not None and hasattr(le, "classes_"):
        try:
            roles += [str(r) for r in le.classes_ if str(r) not in roles]
        except Exception:
            pass
    return sorted(set([r.lower() for r in roles]))

def get_full_roles_for_suggestions() -> List[str]:
    """
    Keep original casing for embeddings display; extended first, then encoder classes.
    """
    roles: List[str] = list(dict.fromkeys(extended_roles))
    _, _, le = get_classic_assets()
    if le is not None and hasattr(le, "classes_"):
        try:
            for r in le.classes_:
                rs = str(r)
                if rs not in roles:
                    roles.append(rs)
        except Exception:
            pass
    return roles


# -----------------
# Text clean helper
# -----------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------
# Prompt parsing utilities (strict filters & requirements)
# -------------------------------------------------------
_EXP_NUM = r"(\d+(?:\.\d+)?)"
_EXPERIENCE_PATTERNS = [
    # between / range
    rf"(?:between|from)\s+{_EXP_NUM}\s*(?:to|and|-|–|—)\s*{_EXP_NUM}\s*(?:years|yrs)\b",
    rf"(?:range|between)\s+{_EXP_NUM}\s*[-–—]\s*{_EXP_NUM}\s*(?:years|yrs)\b",
    rf"(?:min(?:imum)?)[^\d]*{_EXP_NUM}.*?(?:max(?:imum)?)[^\d]*{_EXP_NUM}\s*(?:years|yrs)?\b",
    # comparative
    rf"(?:less than|under|below)\s+{_EXP_NUM}\s*(?:years|yrs)\b",
    rf"(?:greater than|more than|over)\s+{_EXP_NUM}\s*(?:years|yrs)\b",
    rf"(?:at least|min(?:imum)?)\s+{_EXP_NUM}\s*(?:years|yrs)\b",
    # symbols
    rf">=\s*{_EXP_NUM}\s*(?:years|yrs)?\b",
    rf"<=\s*{_EXP_NUM}\s*(?:years|yrs)?\b",
    rf">\s*{_EXP_NUM}\s*(?:years|yrs)?\b",
    rf"<\s*{_EXP_NUM}\s*(?:years|yrs)?\b",
    # simple "3+ years", "7 years"
    rf"{_EXP_NUM}\s*\+\s*(?:years|yrs)\b",
    rf"{_EXP_NUM}\s*(?:years|yrs)\b",
]

_SKILL_SECTION_CLAUSES = [
    r"(?:must include|must have|required(?: skills)?|should include|need to include|need to have)"
]

_LOCATION_PATTERN = r"(?:\bin|from|based in)\s+([a-zA-Z][a-zA-Z\s]{1,50})\b"


def _parse_experience_filters(prompt: str) -> Tuple[Optional[float], Optional[float], Optional[bool], Optional[bool]]:
    """
    Returns (min_years, max_years, strict_ge, strict_le)
    - strict_ge/le indicate if boundary is inclusive for single-sided constraints.
    """
    p = prompt
    # Range patterns first
    m = re.search(rf"(?:between|from)\s+{_EXP_NUM}\s*(?:to|and|-|–|—)\s*{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, None, None
    m = re.search(rf"(?:min(?:imum)?)[^\d]*{_EXP_NUM}.*?(?:max(?:imum)?)[^\d]*{_EXP_NUM}", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, None, None
    m = re.search(rf"{_EXP_NUM}\s*[-–—]\s*{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, None, None

    # One-sided / comparative
    m = re.search(rf"(?:less than|under|below)\s+{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        return None, float(m.group(1)), None, False  # strict <

    m = re.search(rf"(?:greater than|more than|over)\s+{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        return float(m.group(1)) + 1e-9, None, False, None  # strict >

    m = re.search(rf"(?:at least|min(?:imum)?)\s+{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        return float(m.group(1)), None, True, None  # >=

    m = re.search(rf">=\s*{_EXP_NUM}", p)
    if m:
        return float(m.group(1)), None, True, None

    m = re.search(rf"<=\s*{_EXP_NUM}", p)
    if m:
        return None, float(m.group(1)), None, True

    m = re.search(rf">\s*{_EXP_NUM}", p)
    if m:
        return float(m.group(1)) + 1e-9, None, False, None  # strict >

    m = re.search(rf"<\s*{_EXP_NUM}", p)
    if m:
        return None, float(m.group(1)), None, False  # strict <

    # Simple "3+ years"
    m = re.search(rf"{_EXP_NUM}\s*\+\s*(?:years|yrs)\b", p)
    if m:
        return float(m.group(1)), None, True, None

    # Simple "7 years" (treat as >= 7 by default)
    m = re.search(rf"{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        return float(m.group(1)), None, True, None

    return None, None, None, None


def _parse_must_include_skills(prompt: str) -> List[str]:
    """
    Extracts skills after phrases like 'must include', 'must have', 'required skills', etc.
    Splits by comma and 'and'.
    """
    for clause in _SKILL_SECTION_CLAUSES:
        m = re.search(clause + r"\s*([^.;\n]+)", prompt)
        if m:
            span = m.group(1)
            parts = re.split(r",|\/|;| and ", span)
            skills = [clean_text(x).strip() for x in parts if x and clean_text(x).strip()]
            return list(dict.fromkeys([s for s in skills if s]))
    return []


def _parse_projects_required(prompt: str) -> bool:
    return bool(re.search(r"(?:should|must|need to|needs to|required to).{0,10}(?:work on|have)\s+projects", prompt)
                or re.search(r"(?:project experience|projects? section|portfolio of projects)", prompt))


def _parse_location(prompt: str) -> Optional[str]:
    m = re.search(_LOCATION_PATTERN, prompt)
    if m:
        return clean_text(m.group(1)).strip()
    return None


def _parse_role(prompt: str) -> Optional[str]:
    """
    Try to map prompt to a role token present in ROLE_VOCAB.
    If not found but prompt is short (<=5 tokens), fall back to the prompt itself
    so embedding-based role match can still work (e.g., 'advocate').
    """
    tokens = clean_text(prompt).split()
    joined = " ".join(tokens)
    role_vocab = get_role_vocab()  # dynamic, encoder-aware
    # Check multi-word vocab first by length
    for role in sorted(role_vocab, key=lambda x: -len(x)):
        if role in joined:
            return role
    # fallback: early tokens heuristic if prompt looks like a short role-like query
    if len(tokens) <= 5:
        for n in (3, 2, 1):
            cand = " ".join(tokens[:n]).strip()
            if cand in role_vocab:
                return cand
        # NEW: if still not found, allow the short prompt itself (alphabetic only)
        if any(t.isalpha() for t in tokens):
            return joined
    return None


def _role_is_related(predicted_role: str, requested_role: str) -> bool:
    """Semantic proximity using sentence embeddings; threshold tuned modestly."""
    if not predicted_role or not requested_role:
        return False
    pr = clean_text(predicted_role)
    rr = clean_text(requested_role)
    if pr == rr or pr in rr or rr in pr:
        return True
    try:
        emb_a = embedding_model.encode(pr, convert_to_tensor=True)
        emb_b = embedding_model.encode(rr, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b)[0][0].item()
        return sim >= 0.65  # related roles
    except Exception:
        return False


def _role_relatedness_score(predicted_role: str, requested_role: Optional[str]) -> float:
    """
    Return a role match score in [0,1].
    - 1.0 exact/same family
    - ~0.8 related (embedding similarity >= 0.65)
    - 0.0 if unrelated when requested_role present
    - 0.7 neutral default if no requested_role provided
    """
    if not predicted_role:
        return 0.0
    if not requested_role:
        return 0.7  # neutral when user didn't ask a role explicitly
    pr = clean_text(predicted_role)
    rr = clean_text(requested_role)
    if pr == rr or pr in rr or rr in pr:
        return 1.0
    try:
        emb_a = embedding_model.encode(pr, convert_to_tensor=True)
        emb_b = embedding_model.encode(rr, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b)[0][0].item()
        if sim >= 0.75:
            return 0.95
        if sim >= 0.65:
            return 0.8
        return 0.0
    except Exception:
        return 0.0


def _has_projects(projects_field: Any, projects_text: str) -> bool:
    if isinstance(projects_field, list):
        if any(bool(p) for p in projects_field):
            return True
    return "project" in projects_text and len(projects_text) > 20


def _skills_contain_all(candidate_skills: List[str], required: List[str]) -> Tuple[bool, List[str]]:
    cand = set([clean_text(s) for s in candidate_skills if isinstance(s, str)])
    req = [clean_text(s) for s in required if s]
    missing = [s for s in req if s not in cand]
    return (len(missing) == 0), missing


def _experience_satisfies(exp_years: float,
                          min_years: Optional[float],
                          max_years: Optional[float],
                          strict_ge: Optional[bool],
                          strict_le: Optional[bool]) -> bool:
    if min_years is not None:
        if strict_ge is False and not (exp_years > min_years):
            return False
        if strict_ge is True and not (exp_years >= min_years):
            return False
        if strict_ge is None and not (exp_years >= min_years):
            return False
    if max_years is not None:
        if strict_le is False and not (exp_years < max_years):
            return False
        if strict_le is True and not (exp_years <= max_years):
            return False
        if strict_le is None and not (exp_years <= max_years):
            return False
    return True


def _experience_score(exp_years: float,
                      min_years: Optional[float],
                      max_years: Optional[float]) -> float:
    """
    Map experience vs requested bounds to [0,1].
    - 1.0 inside the target range
    - If only min: decays as you fall below
    - If only max: decays as you exceed
    - If no bounds: neutral 0.6
    """
    if min_years is None and max_years is None:
        return 0.6
    # both bounds
    if min_years is not None and max_years is not None:
        if min_years > max_years:
            min_years, max_years = max_years, min_years
        if min_years <= exp_years <= max_years:
            return 1.0
        width = max(1.0, max_years - min_years)
        if exp_years < min_years:
            gap = (min_years - exp_years)
            return max(0.0, 1.0 - gap / (width + 2.0))
        else:
            gap = (exp_years - max_years)
            return max(0.0, 1.0 - gap / (width + 2.0))
    # only min
    if min_years is not None:
        if exp_years >= min_years:
            return 1.0
        gap = (min_years - exp_years)
        base = max(1.0, min_years + 2.0)
        return max(0.0, 1.0 - gap / base)
    # only max
    if exp_years <= max_years:
        return 1.0
    gap = (exp_years - max_years)
    base = max(1.0, max_years + 2.0)
    return max(0.0, 1.0 - gap / base)


def _skills_score(candidate_skills: List[str],
                  must_include: List[str],
                  baseline_required: List[str]) -> float:
    """
    Score skills coverage in [0,1].
    Priority to must_include; if empty, use baseline_required derived from prompt/keywords.
    """
    cand = set([clean_text(s) for s in candidate_skills if isinstance(s, str)])
    focus = [clean_text(s) for s in (must_include if must_include else baseline_required) if s]
    if not focus:
        return 0.6  # neutral if nothing specified
    present = sum(1 for s in focus if s in cand)
    return present / max(1, len(focus))


def _projects_score(projects_field: Any,
                    projects_text: str,
                    projects_required: bool) -> float:
    """
    Score projects presence [0,1].
    """
    has_proj = _has_projects(projects_field, projects_text)
    if projects_required:
        return 1.0 if has_proj else 0.0
    return 0.6 if has_proj else 0.4


# ---------------------------------------------------------
# Semantic matching (owner-scoped + strict/soft filtering)
# ---------------------------------------------------------
async def get_semantic_matches(
    prompt: str,
    threshold: float = 0.45,
    *,
    owner_user_id: Optional[str] = None,
    normalized_prompt: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a ranked list of candidate previews with semantic and shaped scores.
    Now includes strict multi-criteria filtering (experience, skills-must-include,
    projects, location, role) with graceful fallback to close matches if none
    satisfy all constraints. Each item includes flags: is_strict_match, match_type,
    failed_filters (for UI messaging).
    """
    # normalization inputs
    if not normalized_prompt or not isinstance(keywords, list):
        norm = normalize_prompt(prompt)
        normalized_prompt = norm["normalized_prompt"]
        keywords = norm["keywords"]
        is_prompt_role_like = norm.get("is_role_like", len((normalized_prompt or prompt).split()) <= 4)
    else:
        is_prompt_role_like = len((normalized_prompt or prompt).split()) <= 4

    # ❗ Use RAW prompt for parsing filters (digits/symbols preserved)
    raw_prompt = (normalized_prompt or prompt)
    parsed_text = str(raw_prompt).lower()
    # Clean only for embeddings
    cleaned_for_sem = clean_text(raw_prompt)
    prompt_embedding = embedding_model.encode(cleaned_for_sem, convert_to_tensor=True)

    # -----------------------------
    # Parse strict filter directives (on RAW)
    # -----------------------------
    min_years, max_years, strict_ge, strict_le = _parse_experience_filters(parsed_text)
    must_include_skills = _parse_must_include_skills(parsed_text)
    projects_required = _parse_projects_required(parsed_text)
    location_filter = _parse_location(parsed_text)
    requested_role = _parse_role(parsed_text)

    # helpful: do we have any strict filters?
    has_strict_filters = bool(
        requested_role or must_include_skills or projects_required or location_filter or
        (min_years is not None or max_years is not None)
    )

    # quick baseline skills keywords (still used for scoring bonuses)
    baseline_skill_keywords = [
        "react", "aws", "node", "excel", "django", "figma",
        "pandas", "tensorflow", "keras", "java", "python",
        "pytorch", "spark", "sql", "scikit", "mlflow", "docker",
        "kubernetes", "typescript", "next", "nextjs", "next.js", "powerpoint", "flora", "html", "css"
    ]
    baseline_required_skills = sorted(set(
        [kw for kw in (keywords or []) if kw in baseline_skill_keywords] +
        [kw for kw in baseline_skill_keywords if kw in parsed_text]
    ))

    # scope query to owner if provided
    query = {"ownerUserId": owner_user_id} if owner_user_id else {}
    cursor = db.parsed_resumes.find(query)

    strict_matches: List[Dict[str, Any]] = []
    close_matches: List[Dict[str, Any]] = []   # relaxed filters if strict empty AND no strict filters set
    soft_pool: List[Dict[str, Any]] = []       # deep fallback only when no strict filters

    async for cv in cursor:
        cv_id = cv.get("_id")
        raw_text = cv.get("raw_text", "") or ""
        if not raw_text.strip():
            continue

        # Predicted role from prior pipeline (if any)
        original_predicted_role = (cv.get("predicted_role") or cv.get("category") or "").strip()
        predicted_role_lc = original_predicted_role.lower().strip()
        confidence = round(cv.get("confidence", 0), 2)

        cleaned_resume_text = clean_text(raw_text)
        skills = cv.get("skills", []) or []
        skills_text = " ".join(skills).lower()

        # Projects might be list of dicts; make a cheap blob
        projects_field = cv.get("projects", [])
        if isinstance(projects_field, list):
            projects_text = " ".join(
                [p if isinstance(p, str) else (str(p.get("name","")) + " " + str(p.get("description",""))) for p in projects_field]
            ).lower()
        else:
            projects_text = str(projects_field).lower()

        # experience: read multiple aliases if available (robust -> float)
        exp_val = (
            cv.get("total_experience_years")
            or cv.get("years_of_experience")
            or cv.get("experience_years")
            or cv.get("yoe")
            or cv.get("experience")
            or 0
        )
        def _to_float_years(v) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v)
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            return float(m.group(1)) if m else 0.0
        try:
            experience = _to_float_years(exp_val)  # '5+ years' -> 5.0, '0.2 yrs' -> 0.2
        except Exception:
            experience = 0.0

        location_text = clean_text(cv.get("location") or "")

        # Compose comparison text for general semantic sim
        compare_text = predicted_role_lc if is_prompt_role_like else " ".join([
            predicted_role_lc,
            cleaned_resume_text,
            skills_text,
            projects_text,
            f"{experience} years experience",
            location_text,
        ])
        compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
        similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
        semantic_score = round(similarity * 100, 2)

        # -----------------------
        # Compute weighted scoring components
        # -----------------------
        role_score = _role_relatedness_score(predicted_role_lc, requested_role)    # 0..1
        skills_score = _skills_score(skills, must_include_skills, baseline_required_skills)  # 0..1
        exp_score = _experience_score(experience, min_years, max_years)           # 0..1
        proj_score = _projects_score(projects_field, projects_text, projects_required)       # 0..1

        weighted_final = (0.50 * role_score) + (0.25 * skills_score) + (0.15 * exp_score) + (0.10 * proj_score)
        final_score = round(weighted_final * 100.0, 2)  # keep 0..100 like before

        # -------------------------------
        # STRICT FILTER EVALUATION (AND)
        # -------------------------------
        failed_filters: List[str] = []

        # 1) Hard role gate (exact or closely related)
        role_ok = True
        if requested_role:
            role_ok = (role_score >= 0.8)  # exact/related
            if not role_ok:
                failed_filters.append("role")

        # 2) Experience gate
        exp_ok = _experience_satisfies(experience, min_years, max_years, strict_ge, strict_le)
        if (min_years is not None or max_years is not None) and not exp_ok:
            failed_filters.append("experience")

        # 3) Skills must-include (ALL)
        skills_ok = True
        missing_must_skills: List[str] = []
        if must_include_skills:
            skills_ok, missing_must_skills = _skills_contain_all(skills, must_include_skills)
            if not skills_ok:
                failed_filters.append("skills")

        # 4) Projects
        proj_ok = True
        if projects_required:
            proj_ok = _has_projects(projects_field, projects_text)
            if not proj_ok:
                failed_filters.append("projects")

        # 5) Location
        loc_ok = True
        if location_filter:
            loc_ok = location_filter in location_text
            if not loc_ok:
                failed_filters.append("location")

        is_strict_match = (len(failed_filters) == 0)

        # -------------------------------
        # Primary semantic gate (floor)
        # -------------------------------
        primary_gate_pass = True
        if is_prompt_role_like and semantic_score < 50:
            primary_gate_pass = False
        if (not is_prompt_role_like) and semantic_score < 40:
            primary_gate_pass = False

        # -------------------------------
        # Build preview item
        # -------------------------------
        skills_truncated = [s for s in skills[:6] if isinstance(s, str)]
        overflow = max(0, len(skills) - len(skills_truncated))

        item: Dict[str, Any] = {
            "_id": cv_id,
            "name": (cv.get("name") or "").strip() or "No Name",
            "predicted_role": original_predicted_role,
            "category": cv.get("category", original_predicted_role),
            "experience": experience,
            "location": location_text or "N/A",
            "email": cv.get("email", "N/A"),
            "skills": skills,
            "skillsTruncated": skills_truncated,
            "skillsOverflowCount": overflow,
            "resume_url": cv.get("resume_url", ""),
            "semantic_score": semantic_score,
            "confidence": confidence,
            "score_type": "Weighted Role/Skills/Exp/Projects",
            "final_score": final_score,
            "score_components": {
                "role_match": round(role_score, 4),
                "skills_match": round(skills_score, 4),
                "experience_match": round(exp_score, 4),
                "projects_match": round(proj_score, 4),
                "semantic_score": semantic_score
            },
            "rank": 0,
            "related_roles": [],
            "relatedRoles": [],
            "raw_text": raw_text,
            "strengths": [],
            "redFlags": [],
            "is_strict_match": is_strict_match,
            "match_type": "exact" if is_strict_match else "close",
            "failed_filters": failed_filters,
            "mustIncludeSkills": must_include_skills,
            "missingMustSkills": missing_must_skills,
            "matchedSkills": [s for s in skills if isinstance(s, str) and s.lower() in parsed_text],
            "missingSkills": [s for s in (keywords or []) if s not in [x.lower() for x in skills if isinstance(x, str)]],
        }

        # Strengths / Red flags (lightweight)
        if final_score >= 85:
            item["strengths"].append("Strong overall match")
        elif final_score >= 70:
            item["strengths"].append("Relevant to query")
        if requested_role and role_ok:
            item["strengths"].append("Role aligns with requested/related role")
        if not skills:
            item["redFlags"].append("No skills extracted from resume")
        if (min_years is not None or max_years is not None) and not exp_ok:
            item["redFlags"].append("Experience outside requested range")

        # Related roles (best effort)
        try:
            if original_predicted_role:
                pred_embed = embedding_model.encode(original_predicted_role)
                scores = []
                full_roles = get_full_roles_for_suggestions()
                for role in full_roles:
                    role_clean = role.strip().lower()
                    if role_clean == predicted_role_lc:
                        continue
                    role_embed = embedding_model.encode(role)
                    rscore = util.cos_sim(pred_embed, role_embed)[0][0].item()
                    scores.append({"role": role, "match": round(rscore * 100, 2)})
                scores.sort(key=lambda x: x["match"], reverse=True)
                top = scores[:3]
                item["related_roles"] = top
                item["relatedRoles"] = [{"role": x["role"], "score": round(x["match"]/100.0, 4)} for x in top]
        except Exception:
            pass  # non-fatal

        # ----------------------------------------
        # Decide which bucket this item belongs to
        # ----------------------------------------
        # Primary semantic floor
        if not primary_gate_pass:
            # If any strict filter present, ignore weak items entirely
            if has_strict_filters:
                continue
            soft_pool.append(item)
            continue

        # ROLE gate: if role specified and unrelated, drop
        if requested_role and not role_ok:
            continue

        # If strict filters exist, only accept items that pass ALL strict checks
        if has_strict_filters:
            if is_strict_match:
                strict_matches.append(item)
            # else: fail some strict filter -> drop silently
            continue

        # No strict filters -> allow close bucket
        if is_strict_match:
            strict_matches.append(item)
        else:
            close_matches.append(item)

    # -----------------
    # Ranking & return
    # -----------------
    def _rank(items: List[Dict[str, Any]]) -> None:
        items.sort(key=lambda x: (x["final_score"], x.get("semantic_score", 0.0)), reverse=True)
        for i, cv in enumerate(items):
            cv["rank"] = i + 1

    if strict_matches:
        _rank(strict_matches)
        return strict_matches

    if close_matches:
        _rank(close_matches)
        for cv in close_matches:
            cv["is_strict_match"] = False
            cv["match_type"] = "close"
        return close_matches

    # If strict filters were requested but nothing matched -> return empty
    if has_strict_filters:
        return []

    # Otherwise (no strict filters): allow deep fallback
    soft_pool.sort(key=lambda x: x["final_score"], reverse=True)
    for i, cv in enumerate(soft_pool):
        cv["rank"] = i + 1
        cv["is_strict_match"] = False
        cv["match_type"] = "close"
    return soft_pool[:100]


# -----------------------------
# Backwards-compat exports ✅
# -----------------------------
# Some modules (e.g., upload_router) import: model, vectorizer, label_encoder.
# Expose them via lazy loader so those imports keep working without boot failures.
model, vectorizer, label_encoder = get_classic_assets()
