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
# Load classic ML assets (same)
# -----------------------------
MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

model: BaseEstimator = joblib.load(MODEL_PATH)
vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

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

# ✅ Extended real-world roles (priority first)
extended_roles = [
    "AI Engineer", "ML Engineer", "NLP Engineer", "Deep Learning Engineer",
    "MLOps Engineer", "Data Analyst", "Data Engineer", "Computer Vision Engineer",
    "Research Scientist", "Machine Learning Engineer", "Data Scientist",
    "Business Analyst", "Software Engineer", "Full Stack Engineer"
]

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

# Roles vocabulary (lowercased)
ROLE_VOCAB = sorted(set([r.lower() for r in (extended_roles + list(getattr(label_encoder, "classes_", [])))]))


def _parse_experience_filters(prompt: str) -> Tuple[Optional[float], Optional[float], Optional[bool], Optional[bool]]:
    """
    Returns (min_years, max_years, strict_ge, strict_le)
    - strict_ge/le indicate if boundary is inclusive for single-sided constraints.
    """
    p = prompt
    # Range patterns first
    # between X to Y
    m = re.search(rf"(?:between|from)\s+{_EXP_NUM}\s*(?:to|and|-|–|—)\s*{_EXP_NUM}\s*(?:years|yrs)\b", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, None, None
    # min X max Y
    m = re.search(rf"(?:min(?:imum)?)[^\d]*{_EXP_NUM}.*?(?:max(?:imum)?)[^\d]*{_EXP_NUM}", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi, None, None
    # 7-9 yrs
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
            # split by commas and 'and'
            parts = re.split(r",|\/|;| and ", span)
            skills = [clean_text(x).strip() for x in parts if x and clean_text(x).strip()]
            # de-dup and keep non-empty tokens
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
    E.g., 'Find Data Scientist' -> 'data scientist'
    """
    tokens = clean_text(prompt).split()
    joined = " ".join(tokens)
    # Check multi-word vocab first by length
    for role in sorted(ROLE_VOCAB, key=lambda x: -len(x)):
        if role in joined:
            return role
    # fallback: early tokens heuristic if prompt looks like a short role-like query
    if len(tokens) <= 5:
        # try join first 2-3 tokens
        for n in (3, 2, 1):
            cand = " ".join(tokens[:n]).strip()
            if cand in ROLE_VOCAB:
                return cand
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

    cleaned_prompt = clean_text(normalized_prompt or prompt)
    prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)

    # -----------------------------
    # Parse strict filter directives
    # -----------------------------
    min_years, max_years, strict_ge, strict_le = _parse_experience_filters(cleaned_prompt)
    must_include_skills = _parse_must_include_skills(cleaned_prompt)
    projects_required = _parse_projects_required(cleaned_prompt)
    location_filter = _parse_location(cleaned_prompt)
    requested_role = _parse_role(cleaned_prompt)

    # quick baseline skills keywords (still used for scoring bonuses)
    baseline_skill_keywords = [
        "react", "aws", "node", "excel", "django", "figma",
        "pandas", "tensorflow", "keras", "java", "python",
        "pytorch", "spark", "sql", "scikit", "mlflow", "docker",
        "kubernetes", "typescript", "next", "nextjs", "next.js", "powerpoint", "flora"
    ]
    baseline_required_skills = sorted(set(
        [kw for kw in (keywords or []) if kw in baseline_skill_keywords] +
        [kw for kw in baseline_skill_keywords if kw in cleaned_prompt]
    ))

    # scope query to owner if provided
    query = {"ownerUserId": owner_user_id} if owner_user_id else {}
    cursor = db.parsed_resumes.find(query)

    strict_matches: List[Dict[str, Any]] = []
    close_matches: List[Dict[str, Any]] = []   # relaxed filters if strict empty
    soft_pool: List[Dict[str, Any]] = []       # for previous primary-gate fallbacks

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

        # experience: read multiple aliases if available
        exp_val = (
            cv.get("total_experience_years")
            or cv.get("years_of_experience")
            or cv.get("experience_years")
            or cv.get("yoe")
            or cv.get("experience")
            or 0
        )
        try:
            experience = float(exp_val)
        except Exception:
            experience = 0.0

        location_text = clean_text(cv.get("location") or "")

        # Compose comparison text
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
        # Score shaping (no drop)
        # -----------------------
        score = semantic_score

        # Role-like prompts -> demand higher semantic alignment
        if is_prompt_role_like and predicted_role_lc and predicted_role_lc in cleaned_prompt:
            score += 5

        # Skills bonus/penalty (baseline keywords, not strict-must)
        if baseline_required_skills:
            present = sum(1 for s in baseline_required_skills if s in skills_text)
            if present:
                score += min(8, present * 3)  # up to +8
            else:
                score -= 6  # missing all requested (baseline)
        # Location light penalty if doesn't match
        if location_filter:
            if location_filter in location_text:
                score += 5
            else:
                score -= 4

        # Years requirement: bonus if satisfies, penalty if below (for ranking only)
        if (min_years is not None) or (max_years is not None):
            if _experience_satisfies(experience, min_years, max_years, strict_ge, strict_le):
                score += 6
            else:
                # distance-based penalty
                if min_years is not None and experience < min_years:
                    gap = max(0.0, min_years - experience)
                    score -= min(12.0, 4.0 + gap * 2.0)
                if max_years is not None and experience > max_years:
                    gap = max(0.0, experience - max_years)
                    score -= min(8.0, 2.0 + gap * 1.5)

        # Projects hint for ranking
        if _has_projects(projects_field, projects_text):
            score += 3
        elif projects_required:
            score -= 3

        # Clamp score
        score = max(0.0, min(100.0, score))

        # -------------------------------
        # STRICT FILTER EVALUATION (AND)
        # -------------------------------
        failed_filters: List[str] = []

        # 1) Role filter (exact or closely related)
        role_ok = True
        if requested_role:
            role_ok = (predicted_role_lc == requested_role) or _role_is_related(predicted_role_lc, requested_role)
            if not role_ok:
                failed_filters.append("role")

        # 2) Experience filter
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
        # Primary gates (semantic floor)
        # -------------------------------
        primary_gate_pass = True
        if is_prompt_role_like and semantic_score < 50:
            primary_gate_pass = False
        if (not is_prompt_role_like) and semantic_score < 40:
            primary_gate_pass = False

        item = {
            "_id": cv_id,
            "name": (cv.get("name") or "").strip() or "No Name",  # ✅ "No Name" fallback
            "predicted_role": original_predicted_role,
            "category": cv.get("category", original_predicted_role),
            "experience": experience,
            "location": location_text or "N/A",
            "email": cv.get("email", "N/A"),
            "skills": skills,
            "resume_url": cv.get("resume_url", ""),
            "semantic_score": semantic_score,
            "confidence": confidence,
            "score_type": "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content",
            "final_score": round(score, 2),
            "rank": 0,
            "related_roles": [],   # filled below
            "raw_text": raw_text,
            "strengths": [],
            "redFlags": [],
            # For UI messaging and filtering state
            "is_strict_match": is_strict_match,
            "match_type": "exact" if is_strict_match else "close",  # UI can show proper banner
            "failed_filters": failed_filters,
            # Skill matching metadata (based on strict requirement)
            "mustIncludeSkills": must_include_skills,
            "missingMustSkills": missing_must_skills,
            # Contextual skill highlights (vs prompt text)
            "matchedSkills": [s for s in skills if isinstance(s, str) and s.lower() in cleaned_prompt.lower()],
            "missingSkills": [s for s in (keywords or []) if s not in [x.lower() for x in skills if isinstance(x, str)]],
        }

        # Strengths / Red flags
        if semantic_score >= 85:
            item["strengths"].append("Very strong semantic match")
        elif semantic_score >= 70:
            item["strengths"].append("Relevant to prompt")
        if requested_role and _role_is_related(predicted_role_lc, requested_role):
            item["strengths"].append("Role aligns with requested or related role")
        if not skills:
            item["redFlags"].append("No skills extracted from resume")
        if (min_years is not None or max_years is not None) and not exp_ok:
            item["redFlags"].append("Experience outside requested range")

        # Related roles (best effort)
        try:
            if original_predicted_role:
                pred_embed = embedding_model.encode(original_predicted_role)
                scores = []
                full_roles = list(dict.fromkeys(extended_roles + list(label_encoder.classes_)))
                for role in full_roles:
                    role_clean = role.strip().lower()
                    if role_clean == predicted_role_lc:
                        continue
                    role_embed = embedding_model.encode(role)
                    rscore = util.cos_sim(pred_embed, role_embed)[0][0].item()
                    scores.append({"role": role, "match": round(rscore * 100, 2)})
                scores.sort(key=lambda x: x["match"], reverse=True)
                item["related_roles"] = scores[:3]
        except Exception:
            pass  # non-fatal

        # Collect per category
        if primary_gate_pass:
            if is_strict_match:
                strict_matches.append(item)
            else:
                close_matches.append(item)
        else:
            soft_pool.append(item)

    # -----------------
    # Ranking & return
    # -----------------
    def _rank(items: List[Dict[str, Any]]) -> None:
        items.sort(key=lambda x: (x["is_strict_match"], x["final_score"]), reverse=True)
        for i, cv in enumerate(items):
            cv["rank"] = i + 1

    if strict_matches:
        _rank(strict_matches)
        return strict_matches

    # ✅ If no strict matches, fall back to close matches but make it explicit
    if close_matches:
        # Mark match_type explicitly for clarity (already "close")
        _rank(close_matches)
        return close_matches

    # ✅ If nothing passed, use soft_pool (semantic gate fallback) — still marked "close"
    soft_pool.sort(key=lambda x: x["final_score"], reverse=True)
    for i, cv in enumerate(soft_pool):
        cv["rank"] = i + 1
        cv["is_strict_match"] = False
        cv["match_type"] = "close"
    return soft_pool[:100]







# import joblib
# import os
# import re
# import json
# from typing import List, Dict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.base import BaseEstimator
# from sklearn.preprocessing import LabelEncoder
# from app.utils.mongo import db
# from sentence_transformers import SentenceTransformer, util

# # Load models
# MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
# VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
# ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

# model: BaseEstimator = joblib.load(MODEL_PATH)
# vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
# label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# # Load semantic model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Load dynamic semantic role map from file
# SEMANTIC_ROLE_MAP_PATH = os.path.join("app", "semantic_role_pool.json")
# try:
#     with open(SEMANTIC_ROLE_MAP_PATH, "r", encoding="utf-8") as f:
#         semantic_role_map = json.load(f)
# except Exception as e:
#     print(f"[!] Failed to load semantic role pool: {e}")
#     semantic_role_map = {}

# # Clean text utility
# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"\S+@\S+", '', text)
#     text = re.sub(r"@\w+|#", '', text)
#     text = re.sub(r"[^a-zA-Z\s]", ' ', text)
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# # ✅ Hybrid filtering with prompt + predicted role + related roles
# async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
#     cleaned_prompt = clean_text(prompt)
#     prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)
#     is_prompt_role_like = len(cleaned_prompt.split()) <= 4

#     # Extract filters
#     years_req = 0
#     required_skills = []
#     location_filter = ""
#     project_required = False

#     exp_patterns = [
#         r"(?:minimum|at least|more than)\s*(\d+)\s*(?:years|yrs)",
#         r"(\d+)\s*\+?\s*(?:years|yrs)"
#     ]
#     for pattern in exp_patterns:
#         match = re.search(pattern, cleaned_prompt)
#         if match:
#             years_req = int(match.group(1))
#             if "more than" in cleaned_prompt:
#                 years_req += 1
#             break

#     loc_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]+)", cleaned_prompt)
#     if loc_match:
#         location_filter = loc_match.group(1).strip().lower()

#     if any(kw in cleaned_prompt for kw in ["project", "worked on", "contributed to"]):
#         project_required = True

#     skill_keywords = ["react", "aws", "node", "excel", "django", "figma", "pandas", "tensorflow", "keras", "java", "python"]
#     required_skills = [kw for kw in skill_keywords if kw in cleaned_prompt]

#     all_roles = list(label_encoder.classes_)

#     matched = []
#     seen_ids = set()
#     cursor = db.parsed_resumes.find({})

#     async for cv in cursor:
#         cv_id = cv.get("_id")
#         if not cv_id or cv_id in seen_ids:
#             continue
#         seen_ids.add(cv_id)

#         raw_text = cv.get("raw_text", "")
#         if not raw_text.strip():
#             continue

#         original_predicted_role = cv.get("predicted_role") or ""
#         predicted_role = original_predicted_role.lower().strip()
#         confidence = round(cv.get("confidence", 0), 2)
#         cleaned_resume_text = clean_text(raw_text)

#         skills = cv.get("skills", [])
#         skills_text = " ".join(skills)
#         projects = cv.get("projects", []) if isinstance(cv.get("projects"), list) else []
#         projects_text = " ".join(projects)
#         experience = cv.get("experience", 0)
#         experience_text = f"{experience} years experience"
#         location_text = (cv.get("location") or "").lower()

#         if years_req > 0 and experience < years_req:
#             continue
#         if required_skills and not any(skill.lower() in skills_text.lower() for skill in required_skills):
#             continue
#         if location_filter and location_filter not in location_text:
#             continue
#         if project_required and not any("project" in proj.lower() or len(proj) > 30 for proj in projects):
#             continue

#         compare_text = predicted_role if is_prompt_role_like else " ".join([
#             predicted_role,
#             cleaned_resume_text,
#             skills_text,
#             projects_text,
#             experience_text,
#             location_text
#         ])

#         compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
#         similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
#         semantic_score = round(similarity * 100, 2)

#         if is_prompt_role_like and semantic_score < 70:
#             continue
#         if not is_prompt_role_like and semantic_score < 60:
#             continue

#         # ✅ Related role prediction from JSON map
#         related_roles = []
#         try:
#             related_roles = semantic_role_map.get(original_predicted_role, [])
#         except Exception as e:
#             print(f"[!] Related role lookup failed: {e}")

#         final_score = semantic_score
#         score_type = "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content"

#         matched.append({
#             "_id": cv_id,
#             "name": cv.get("name") or "Unnamed",
#             "predicted_role": original_predicted_role,
#             "experience": experience,
#             "location": location_text or "N/A",
#             "email": cv.get("email", "N/A"),
#             "skills": skills,
#             "resume_url": cv.get("resume_url", ""),
#             "semantic_score": semantic_score,
#             "confidence": confidence,
#             "score_type": score_type,
#             "final_score": final_score,
#             "related_roles": related_roles,
#             "rank": 0,
#             "raw_text": raw_text
#         })

#     matched.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, cv in enumerate(matched):
#         cv["rank"] = i + 1

#     return matched


# import joblib
# import os
# import re
# from typing import List, Dict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.base import BaseEstimator
# from sklearn.preprocessing import LabelEncoder
# from app.utils.mongo import db
# from sentence_transformers import SentenceTransformer, util

# # Load models
# MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
# VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
# ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

# model: BaseEstimator = joblib.load(MODEL_PATH)
# vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
# label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# # Load semantic model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Extended real-world roles (beyond label encoder)
# extended_roles = [
#     "AI Engineer", "ML Engineer", "NLP Engineer", "Deep Learning Engineer",
#     "MLOps Engineer", "Data Analyst", "Data Engineer", "Computer Vision Engineer",
#     "Research Scientist", "Machine Learning Engineer"
# ]

# # Clean text utility
# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"\S+@\S+", '', text)
#     text = re.sub(r"@\w+|#", '', text)
#     text = re.sub(r"[^a-zA-Z\s]", ' ', text)
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# # ✅ Hybrid filtering with prompt + predicted role + related roles
# async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
#     cleaned_prompt = clean_text(prompt)
#     prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)
#     is_prompt_role_like = len(cleaned_prompt.split()) <= 4

#     # Extract filters
#     years_req = 0
#     required_skills = []
#     location_filter = ""
#     project_required = False

#     exp_patterns = [
#         r"(?:minimum|at least|more than)\s*(\d+)\s*(?:years|yrs)",
#         r"(\d+)\s*\+?\s*(?:years|yrs)"
#     ]
#     for pattern in exp_patterns:
#         match = re.search(pattern, cleaned_prompt)
#         if match:
#             years_req = int(match.group(1))
#             if "more than" in cleaned_prompt:
#                 years_req += 1
#             break

#     loc_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]+)", cleaned_prompt)
#     if loc_match:
#         location_filter = loc_match.group(1).strip().lower()

#     if any(kw in cleaned_prompt for kw in ["project", "worked on", "contributed to"]):
#         project_required = True

#     skill_keywords = ["react", "aws", "node", "excel", "django", "figma", "pandas", "tensorflow", "keras", "java", "python"]
#     required_skills = [kw for kw in skill_keywords if kw in cleaned_prompt]

#     all_roles = list(label_encoder.classes_)
#     matched = []
#     seen_ids = set()
#     cursor = db.parsed_resumes.find({})

#     async for cv in cursor:
#         cv_id = cv.get("_id")
#         if not cv_id or cv_id in seen_ids:
#             continue
#         seen_ids.add(cv_id)

#         raw_text = cv.get("raw_text", "")
#         if not raw_text.strip():
#             continue

#         original_predicted_role = cv.get("predicted_role") or ""
#         predicted_role = original_predicted_role.lower().strip()
#         confidence = round(cv.get("confidence", 0), 2)
#         cleaned_resume_text = clean_text(raw_text)

#         skills = cv.get("skills", [])
#         skills_text = " ".join(skills)
#         projects = cv.get("projects", []) if isinstance(cv.get("projects"), list) else []
#         projects_text = " ".join(projects)
#         experience = cv.get("experience", 0)
#         experience_text = f"{experience} years experience"
#         location_text = (cv.get("location") or "").lower()

#         if years_req > 0 and experience < years_req:
#             continue
#         if required_skills and not any(skill.lower() in skills_text.lower() for skill in required_skills):
#             continue
#         if location_filter and location_filter not in location_text:
#             continue
#         if project_required and not any("project" in proj.lower() or len(proj) > 30 for proj in projects):
#             continue

#         compare_text = predicted_role if is_prompt_role_like else " ".join([
#             predicted_role,
#             cleaned_resume_text,
#             skills_text,
#             projects_text,
#             experience_text,
#             location_text
#         ])

#         compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
#         similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
#         semantic_score = round(similarity * 100, 2)

#         if is_prompt_role_like and semantic_score < 70:
#             continue
#         if not is_prompt_role_like and semantic_score < 60:
#             continue

#         # ✅ Real-time cosine similarity-based related role generation (Top 3 only)
#         related_roles = []
#         if original_predicted_role.strip():
#             try:
#                 pred_embed = embedding_model.encode(original_predicted_role)
#                 scores = []

#                 full_roles = list(set(all_roles + extended_roles))

#                 for role in full_roles:
#                     role_clean = role.strip().lower()
#                     if role_clean == predicted_role:
#                         continue
#                     role_embed = embedding_model.encode(role)
#                     score = util.cos_sim(pred_embed, role_embed)[0][0].item()
#                     scores.append({
#                         "role": role,
#                         "match": round(score * 100, 2)
#                     })

#                 scores.sort(key=lambda x: x["match"], reverse=True)
#                 related_roles = scores[:3]
#             except Exception as e:
#                 print(f"[!] Real-time related role generation failed: {e}")

#         final_score = semantic_score
#         score_type = "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content"

#         matched.append({
#             "_id": cv_id,
#             "name": cv.get("name") or "Unnamed",
#             "predicted_role": original_predicted_role,
#             "experience": experience,
#             "location": location_text or "N/A",
#             "email": cv.get("email", "N/A"),
#             "skills": skills,
#             "resume_url": cv.get("resume_url", ""),
#             "semantic_score": semantic_score,
#             "confidence": confidence,
#             "score_type": score_type,
#             "final_score": final_score,
#             "related_roles": related_roles,
#             "rank": 0,
#             "raw_text": raw_text
#         })

#     matched.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, cv in enumerate(matched):
#         cv["rank"] = i + 1

#     return matched
