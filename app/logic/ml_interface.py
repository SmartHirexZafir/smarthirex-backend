import joblib
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from app.utils.mongo import db
from sentence_transformers import SentenceTransformer, util
from app.logic.normalize import normalize_prompt

# NEW: small extras for ANN
import math
from bson import ObjectId
try:
    import numpy as np
except Exception:
    np = None  # degrade gracefully if numpy unavailable

# NEW: for near-literal paragraph matching
from difflib import SequenceMatcher
# NEW: timestamps for embedding persistence
from datetime import datetime, timezone


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
    "Graphic Designer", "Visual Designer","Python Developer"
]

# 🔹 ADDITIVE: Extend with legal roles (won't break existing behavior)
extended_roles += [
    "Advocate", "Lawyer", "Attorney", "Legal Counsel", "Corporate Counsel",
    "Associate Lawyer", "Litigation Lawyer"
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


# =================================================================
# NEW: Config loader for weights/thresholds (search_weights.json)
# =================================================================
SEARCH_WEIGHTS_PATH = os.path.join("app", "resources", "search_weights.json")

_DEFAULT_SEARCH_CFG: Dict[str, Any] = {
    "weights": {
        "role": 0.50,
        "skills": 0.25,
        "experience": 0.15,
        "projects": 0.10
    },
    "thresholds": {
        "semantic_floor_role_like": 50,
        "semantic_floor_general": 40,
        "paragraph_floor": 35
    },
    "boosts": {
        "literal_semantic": 99.5,
        "near_literal_semantic": 92.0,
        "role_similarity_strict": 0.80,
        "role_similarity_high": 0.75,
        "role_similarity_related": 0.65
    },
    "ann": {
        "use_ann_index": True,
        "topk": 800,
        "emb_batch_size": 128,
        "truncate_chars": 4000
    },
    "baseline_skill_keywords": [
        "react", "aws", "node", "excel", "django", "figma",
        "pandas", "tensorflow", "keras", "java", "python",
        "pytorch", "spark", "sql", "scikit", "mlflow", "docker",
        "kubernetes", "typescript", "next", "nextjs", "next.js",
        "powerpoint", "flora", "html", "css"
    ]
}

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_search_cfg() -> Dict[str, Any]:
    cfg = dict(_DEFAULT_SEARCH_CFG)
    # deep copy nested dicts
    cfg["weights"] = dict(_DEFAULT_SEARCH_CFG["weights"])
    cfg["thresholds"] = dict(_DEFAULT_SEARCH_CFG["thresholds"])
    cfg["boosts"] = dict(_DEFAULT_SEARCH_CFG["boosts"])
    cfg["ann"] = dict(_DEFAULT_SEARCH_CFG["ann"])
    cfg["baseline_skill_keywords"] = list(_DEFAULT_SEARCH_CFG["baseline_skill_keywords"])
    try:
        if os.path.exists(SEARCH_WEIGHTS_PATH):
            with open(SEARCH_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
                if isinstance(file_cfg, dict):
                    _deep_update(cfg, file_cfg)
    except Exception:
        # optional file; ignore errors
        pass

    # Env overrides (preserve old behavior)
    def _env_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val not in ("0", "false", "False", "no", "NO")

    def _env_int(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return int(val)
        except Exception:
            return default

    def _env_float(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    # ANN/env
    cfg["ann"]["use_ann_index"] = _env_bool("USE_ANN_INDEX", bool(cfg["ann"]["use_ann_index"]))
    cfg["ann"]["topk"] = _env_int("ANN_TOPK", int(cfg["ann"]["topk"]))
    cfg["ann"]["emb_batch_size"] = _env_int("EMB_BATCH_SIZE", int(cfg["ann"]["emb_batch_size"]))
    cfg["ann"]["truncate_chars"] = _env_int("INDEX_TRUNCATE_CHARS", int(cfg["ann"]["truncate_chars"]))

    # thresholds/env
    th = cfg["thresholds"]
    th["semantic_floor_role_like"] = _env_float("SEMANTIC_FLOOR_ROLE_LIKE", float(th["semantic_floor_role_like"]))
    th["semantic_floor_general"] = _env_float("SEMANTIC_FLOOR_GENERAL", float(th["semantic_floor_general"]))
    th["paragraph_floor"] = _env_float("PARAGRAPH_FLOOR", float(th["paragraph_floor"]))

    # boosts/env
    bs = cfg["boosts"]
    bs["literal_semantic"] = _env_float("LITERAL_SEMANTIC_BOOST", float(bs["literal_semantic"]))
    bs["near_literal_semantic"] = _env_float("NEAR_LITERAL_SEMANTIC_BOOST", float(bs["near_literal_semantic"]))
    bs["role_similarity_strict"] = _env_float("ROLE_SIM_STRICT", float(bs["role_similarity_strict"]))
    bs["role_similarity_high"] = _env_float("ROLE_SIM_HIGH", float(bs["role_similarity_high"]))
    bs["role_similarity_related"] = _env_float("ROLE_SIM_RELATED", float(bs["role_similarity_related"]))

    # weights/env
    ws = cfg["weights"]
    ws["role"] = _env_float("WEIGHT_ROLE", float(ws["role"]))
    ws["skills"] = _env_float("WEIGHT_SKILLS", float(ws["skills"]))
    ws["experience"] = _env_float("WEIGHT_EXPERIENCE", float(ws["experience"]))
    ws["projects"] = _env_float("WEIGHT_PROJECTS", float(ws["projects"]))

    return cfg

_SEARCH_CFG = _load_search_cfg()

# expose concrete values
W_ROLE = float(_SEARCH_CFG["weights"]["role"])
W_SKILLS = float(_SEARCH_CFG["weights"]["skills"])
W_EXP = float(_SEARCH_CFG["weights"]["experience"])
W_PROJ = float(_SEARCH_CFG["weights"]["projects"])

SEMANTIC_FLOOR_ROLE_LIKE = float(_SEARCH_CFG["thresholds"]["semantic_floor_role_like"])
SEMANTIC_FLOOR_GENERAL = float(_SEARCH_CFG["thresholds"]["semantic_floor_general"])
PARAGRAPH_FLOOR = float(_SEARCH_CFG["thresholds"]["paragraph_floor"])

LITERAL_SEMANTIC_BOOST = float(_SEARCH_CFG["boosts"]["literal_semantic"])
NEAR_LITERAL_SEMANTIC_BOOST = float(_SEARCH_CFG["boosts"]["near_literal_semantic"])
ROLE_SIM_STRICT = float(_SEARCH_CFG["boosts"]["role_similarity_strict"])
ROLE_SIM_HIGH = float(_SEARCH_CFG["boosts"]["role_similarity_high"])
ROLE_SIM_RELATED = float(_SEARCH_CFG["boosts"]["role_similarity_related"])

BASELINE_SKILL_KEYWORDS = list(_SEARCH_CFG.get("baseline_skill_keywords", []))

# ============================================================
# NEW: Simple owner-scoped ANN/Vector index (lazy, in-memory)
# ============================================================

# Config via config/env (safe)
USE_ANN_INDEX = bool(_SEARCH_CFG["ann"]["use_ann_index"])
ANN_TOPK = int(_SEARCH_CFG["ann"]["topk"])
EMB_BATCH_SIZE = int(_SEARCH_CFG["ann"]["emb_batch_size"])
INDEX_TRUNCATE_CHARS = int(_SEARCH_CFG["ann"]["truncate_chars"])

# Global cache: owner -> {"ids": [str], "vecs": numpy.ndarray(N,d)}
_owner_index_cache: Dict[str, Dict[str, Any]] = {}  # key "__all__" used when owner_user_id is None

def _owner_key(owner_user_id: Optional[str]) -> str:
    return owner_user_id or "__all__"

def _l2_normalize(mat: "np.ndarray") -> "np.ndarray":
    if np is None:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _to_object_id(maybe_id: Any) -> Optional[ObjectId]:
    try:
        return ObjectId(maybe_id)
    except Exception:
        return None

def _build_search_blob(cv: Dict[str, Any]) -> str:
    """
    Build a compact, generic text for ANN indexing. Uses cheap fields only.
    Falls back to raw_text (truncated). Safe for all docs.
    """
    parts: List[str] = []
    pr = str(cv.get("predicted_role") or cv.get("category") or "").strip()
    nm = str(cv.get("name") or "").strip()
    loc = str(cv.get("location") or "").strip()
    exp = cv.get("total_experience_years") or cv.get("years_of_experience") or cv.get("experience_years") or cv.get("yoe") or cv.get("experience")

    if pr: parts.append(pr)
    if nm: parts.append(nm)
    if loc: parts.append(loc)

    skills = cv.get("skills") or []
    if isinstance(skills, list) and skills:
        parts.append(" ".join([str(s) for s in skills if s]))

    projects = cv.get("projects") or []
    if isinstance(projects, list) and projects:
        proj_bits = []
        for p in projects[:6]:
            if isinstance(p, str):
                proj_bits.append(p)
            elif isinstance(p, dict):
                proj_bits.append(str(p.get("name") or ""))
                proj_bits.append(str(p.get("description") or ""))
        if proj_bits:
            parts.append(" ".join([x for x in proj_bits if x]))

    if exp:
        parts.append(str(exp))

    raw_text = str(cv.get("raw_text") or "")
    if raw_text:
        parts.append(raw_text[:INDEX_TRUNCATE_CHARS])  # cap

    return clean_text(" ".join([p for p in parts if p]).strip())

async def _ensure_owner_index(owner_user_id: Optional[str]) -> None:
    """
    Build (or reuse) an owner-scoped vector index lazily.
    Prefers precomputed `index_embedding` stored on each resume; if missing,
    computes once and persists back for future fast boots. In-memory cache only.
    """
    if not USE_ANN_INDEX or np is None:
        return
    key = _owner_key(owner_user_id)
    if key in _owner_index_cache and _owner_index_cache[key].get("vecs") is not None:
        return

    query = {"ownerUserId": owner_user_id} if owner_user_id else {}
    # fetch minimal fields including stored embeddings
    projection = {
        "_id": 1,
        "index_embedding": 1,
        "index_embedding_dim": 1,
        "index_embedding_model": 1,
        "index_blob": 1,
        "raw_text": 1,
        "predicted_role": 1,
        "category": 1,
        "name": 1,
        "location": 1,
        "skills": 1,
        "projects": 1,
        "total_experience_years": 1,
        "years_of_experience": 1,
        "experience_years": 1,
        "yoe": 1,
        "experience": 1
    }

    docs: List[Dict[str, Any]] = await db.parsed_resumes.find(query, projection).to_list(length=100_000)
    if not docs:
        _owner_index_cache[key] = {"ids": [], "vecs": None}
        return

    ids: List[str] = []
    vecs: List["np.ndarray"] = []

    for cv in docs:
        rid = str(cv.get("_id"))
        stored = cv.get("index_embedding")
        if stored is not None:
            try:
                vec_np = np.asarray(stored, dtype=np.float32)
                n = np.linalg.norm(vec_np) + 1e-12
                vec_np = vec_np / n
            except Exception:
                vec_np = None
        else:
            vec_np = None

        if vec_np is None:
            # compute from a compact blob
            blob = cv.get("index_blob") or _build_search_blob(cv)
            try:
                emb = embedding_model.encode(clean_text(blob)[:INDEX_TRUNCATE_CHARS], convert_to_tensor=False)
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                emb = emb.astype(np.float32)
                emb /= (np.linalg.norm(emb) + 1e-12)
                vec_np = emb

                # persist for future fast warm
                try:
                    await db.parsed_resumes.update_one(
                        {"_id": cv["_id"]},
                        {"$set": {
                            "index_blob": blob,
                            "index_embedding": emb.tolist(),
                            "index_embedding_dim": int(emb.shape[0]),
                            "index_embedding_model": str(MODEL_ID),
                            "ann_ready": True,
                            "index_embedding_updated_at": datetime.now(timezone.utc)
                        }}
                    )
                except Exception:
                    # best-effort; ignore persistence failures
                    pass
            except Exception:
                # could not compute, skip this doc for ANN
                vec_np = None

        if vec_np is not None:
            ids.append(rid)
            vecs.append(vec_np)

    if not vecs:
        _owner_index_cache[key] = {"ids": [], "vecs": None}
        return

    try:
        mat = np.vstack(vecs).astype(np.float32)
    except Exception:
        _owner_index_cache[key] = {"ids": [], "vecs": None}
        return

    _owner_index_cache[key] = {"ids": ids, "vecs": mat}

def _index_search_ids(owner_user_id: Optional[str], q_vec_np: "np.ndarray", topk: int) -> List[str]:
    """
    Return top-k resume _id strings from the cached index (if present).
    If index missing/unavailable, returns [] to signal fallback.
    """
    if not USE_ANN_INDEX or np is None:
        return []
    key = _owner_key(owner_user_id)
    state = _owner_index_cache.get(key)
    if not state or state.get("vecs") is None:
        return []
    ids: List[str] = state["ids"]
    vecs: np.ndarray = state["vecs"]
    if vecs is None or vecs.shape[0] == 0:
        return []

    # Cosine via dot (vectors are L2-normalized)
    try:
        sims = vecs @ q_vec_np  # shape (N,)
        topk = max(1, min(topk, sims.shape[0]))
        idx = np.argpartition(sims, -topk)[-topk:]
        # sort those topk by descending score
        idx = idx[np.argsort(-sims[idx])]
        return [ids[i] for i in idx.tolist()]
    except Exception:
        return []

def add_to_index(owner_user_id: Optional[str], resume_id: Any, text_blob: str) -> None:
    """
    Optional helper for ingestion path: add/update one resume into the in-memory index.
    Safe no-op if ANN disabled or numpy missing.
    """
    if not USE_ANN_INDEX or np is None:
        return
    key = _owner_key(owner_user_id)
    state = _owner_index_cache.get(key)
    if state is None or state.get("vecs") is None:
        # index not built yet; ignore (it will be built lazily on first query)
        return
    try:
        vec = embedding_model.encode(clean_text(text_blob)[:INDEX_TRUNCATE_CHARS],
                                     convert_to_tensor=False)
        if isinstance(vec, list):
            vec = np.array(vec, dtype=np.float32)
        vec = vec.astype(np.float32)
        # normalize
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
        # update/append
        rid = str(resume_id)
        if rid in state["ids"]:
            i = state["ids"].index(rid)
            state["vecs"][i, :] = vec
        else:
            state["ids"].append(rid)
            state["vecs"] = np.vstack([state["vecs"], vec[None, :]])
    except Exception:
        # silent no-op on failure
        pass

# ============================================================
# END: ANN / Vector index helpers
# ============================================================


# ---------------------------------------------------------
# Helpers for strong paragraph matching (additive, non-breaking)
# ---------------------------------------------------------
def _normalize_blob_for_literal(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _looks_like_paragraph(q: str) -> bool:
    q = (q or "").strip()
    # Long text OR many tokens OR contains quotes → treat as paragraph
    return (len(q) >= 60) or ('"' in q) or ("'" in q) or (len(q.split()) >= 12)


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

    NEW: Uses a lazy owner-scoped ANN/Vector index to pull a small candidate pool
    (top-K) before running the existing filtering pipeline. If ANN is not
    available, behavior gracefully falls back to the previous full-scan approach.

    NEW (strong filter): For paragraph-like queries, perform literal or near-literal
    matching against resume raw_text and boost those hits so exact passages are surfaced.
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
    prompt_embedding_tensor = embedding_model.encode(cleaned_for_sem, convert_to_tensor=True)

    # Strong filter setup (paragraph-literal features)
    query_raw_for_literal = (normalized_prompt or prompt or "").strip()
    query_lit = _normalize_blob_for_literal(query_raw_for_literal)
    is_paragraph_query = _looks_like_paragraph(query_raw_for_literal)

    # Also keep a numpy vector for ANN (if possible)
    if np is not None:
        try:
            q_vec_np = embedding_model.encode(cleaned_for_sem, convert_to_tensor=False)
            if isinstance(q_vec_np, list):
                q_vec_np = np.array(q_vec_np, dtype=np.float32)
            # normalize for cosine via dot
            nrm = np.linalg.norm(q_vec_np) + 1e-12
            q_vec_np = (q_vec_np / nrm).astype(np.float32)
        except Exception:
            q_vec_np = None
    else:
        q_vec_np = None

    # -----------------------------
    # Parse strict filter directives (on RAW)
    # -----------------------------
    min_years, max_years, strict_ge, strict_le = _parse_experience_filters(parsed_text)
    must_include_skills = _parse_must_include_skills(parsed_text)
    projects_required = _parse_projects_required(parsed_text)
    location_filter = _parse_location(parsed_text)
    requested_role = _parse_role(parsed_text)

    # NEW: education & phrases
    schools_required, degrees_required = _parse_education_requirements(raw_prompt)
    must_have_phrases, exclude_phrases = _parse_phrase_requirements(raw_prompt)

    # helpful: do we have any strict filters?
    has_strict_filters = bool(
        requested_role or must_include_skills or projects_required or location_filter or
        schools_required or degrees_required or must_have_phrases or exclude_phrases or
        (min_years is not None or max_years is not None)
    )

    # quick baseline skills keywords (still used for scoring bonuses)
    baseline_skill_keywords = BASELINE_SKILL_KEYWORDS or [
        "react", "aws", "node", "excel", "django", "figma",
        "pandas", "tensorflow", "keras", "java", "python",
        "pytorch", "spark", "sql", "scikit", "mlflow", "docker",
        "kubernetes", "typescript", "next", "nextjs", "next.js", "powerpoint", "flora", "html", "css"
    ]
    baseline_required_skills = sorted(set(
        [kw for kw in (keywords or []) if kw in baseline_skill_keywords] +
        [kw for kw in baseline_skill_keywords if kw in parsed_text]
    ))

    # =====================================================
    # NEW: ANN candidate recall (owner-scoped, lazy index)
    # =====================================================
    ids_restrict: Optional[List[str]] = None
    if USE_ANN_INDEX and q_vec_np is not None and np is not None:
        # build index first time for this owner (no-op if exists)
        await _ensure_owner_index(owner_user_id)
        try:
            ids_restrict = _index_search_ids(owner_user_id, q_vec_np, ANN_TOPK)
            # if empty list, we simply don't restrict later (full fallback)
            if ids_restrict is not None and len(ids_restrict) == 0:
                ids_restrict = None
        except Exception:
            ids_restrict = None

    # scope query to owner if provided; restrict to ANN candidate IDs when available
    query: Dict[str, Any] = {"ownerUserId": owner_user_id} if owner_user_id else {}
    if ids_restrict:
        # Convert to ObjectIds where possible; also keep string IDs for robustness
        obj_ids = [oid for oid in ([_to_object_id(i) for i in ids_restrict]) if oid is not None]
        query["_id"] = {"$in": obj_ids or ids_restrict}

    # NEW: prefilter by min_years if present (cheap DB-side pruning)
    if min_years is not None:
        query["$or"] = [
            {"total_experience_years": {"$gte": min_years}},
            {"years_of_experience": {"$gte": min_years}},
            {"experience_years": {"$gte": min_years}},
            {"yoe": {"$gte": min_years}},
            {"experience": {"$gte": min_years}},
        ]

    # Minimal projection to reduce network/load
    projection = {
        "_id": 1,
        "raw_text": 1,
        "predicted_role": 1,
        "category": 1,
        "name": 1,
        "location": 1,
        "email": 1,
        "skills": 1,
        "projects": 1,
        "resume_url": 1,
        "confidence": 1,
        "total_experience_years": 1,
        "years_of_experience": 1,
        "experience_years": 1,
        "yoe": 1,
        "experience": 1,
    }

    cursor = db.parsed_resumes.find(query, projection=projection)

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

        # ---- Add: user-friendly experience fields for UI ----
        experience_rounded = round(experience, 2)
        if experience > 0 and experience < 1:
            experience_display = "< 1 year"
        elif experience >= 1:
            yrs_i = int(round(experience))
            experience_display = f"{yrs_i} year" + ("" if yrs_i == 1 else "s")
        else:
            experience_display = "Not specified"
        # -----------------------------------------------------

        location_text = clean_text(cv.get("location") or "")

        # Compose comparison text for general semantic sim
        # (Cosmetic: format experience cleanly to avoid "0." artifacts)
        exp_str = f"{experience:.1f}".rstrip('0').rstrip('.')
        if exp_str == "":
            exp_str = "0"
        compare_text = predicted_role_lc if is_prompt_role_like else " ".join([
            predicted_role_lc,
            cleaned_resume_text,
            skills_text,
            projects_text,
            f"{exp_str} years experience",
            location_text,
        ])
        # (Keep original behavior: per-item encode for semantic_score; safe if ANN already shrank pool)
        compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
        similarity = util.cos_sim(prompt_embedding_tensor, compare_embedding)[0][0].item()
        semantic_score = round(similarity * 100, 2)

        # -----------------------
        # Strong paragraph matching (additive)
        # -----------------------
        literal_hit = False
        near_literal_hit = False

        # We keep your original paragraph gate intact,
        # but ALSO allow smaller snippets (>=20 chars) to attempt literal checks.
        allow_literal_attempt = is_paragraph_query or (len(query_lit) >= 20)

        if allow_literal_attempt and raw_text:
            cv_blob = _normalize_blob_for_literal(raw_text)
            # 1) exact/substring literal check
            if query_lit and query_lit in cv_blob:
                literal_hit = True
            else:
                # 2) near-literal (cheap sliding window) for longer snippets
                q_snip = query_lit[:240]
                if len(q_snip) >= 60:
                    step = 800  # coarse step to keep it light
                    for start in range(0, max(1, len(cv_blob) - len(q_snip) + 1), step):
                        window = cv_blob[start:start + len(q_snip) + 200]
                        if not window:
                            continue
                        sim = SequenceMatcher(None, q_snip, window).ratio()
                        if sim >= 0.85:
                            near_literal_hit = True
                            break

        # -----------------------
        # Compute weighted scoring components
        # -----------------------
        role_score = _role_relatedness_score(predicted_role_lc, requested_role)    # 0..1
        skills_score = _skills_score(skills, must_include_skills, baseline_required_skills)  # 0..1
        exp_score = _experience_score(experience, min_years, max_years)           # 0..1
        proj_score = _projects_score(projects_field, projects_text, projects_required)       # 0..1

        # Boosts on literal/near-literal (without breaking existing weights)
        if literal_hit:
            semantic_score = max(semantic_score, LITERAL_SEMANTIC_BOOST)
            # If role was neutral and user asked a role, gently lift
            if requested_role:
                role_score = max(role_score, 0.90)
        elif near_literal_hit:
            semantic_score = max(semantic_score, NEAR_LITERAL_SEMANTIC_BOOST)

        weighted_final = (W_ROLE * role_score) + (W_SKILLS * skills_score) + (W_EXP * exp_score) + (W_PROJ * proj_score)
        final_score = round(weighted_final * 100.0, 2)  # keep 0..100 like before

        # -------------------------------
        # STRICT FILTER EVALUATION (AND)
        # -------------------------------
        failed_filters: List[str] = []

        # 1) Hard role gate (exact or closely related)
        role_ok = True
        if requested_role:
            role_ok = (role_score >= ROLE_SIM_STRICT)  # exact/related
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

        # 6) Education (schools & degrees) — strict contains in raw_text
        edu_ok = True
        rt_lc = (raw_text or "").lower()
        if schools_required:
            edu_ok = all(s in rt_lc for s in schools_required)
            if not edu_ok:
                failed_filters.append("education_schools")
        if edu_ok and degrees_required:
            # accept common degree variants like dots/hyphens removed
            norm_rt = re.sub(r"[.\- ]", "", rt_lc)
            def _deg_match(d: str) -> bool:
                d1 = d.lower()
                d2 = re.sub(r"[.\- ]", "", d1)
                return (d1 in rt_lc) or (d2 in norm_rt)
            deg_ok = all(_deg_match(d) for d in degrees_required)
            if not deg_ok:
                failed_filters.append("education_degrees")
                edu_ok = False

        # 7) Phrases include/exclude — strict substring on raw_text
        phr_ok = True
        if must_have_phrases:
            if not all(p.lower() in rt_lc for p in must_have_phrases):
                failed_filters.append("phrases_required")
                phr_ok = False
        if exclude_phrases and phr_ok:
            if any(p.lower() in rt_lc for p in exclude_phrases):
                failed_filters.append("phrases_excluded")
                phr_ok = False

        is_strict_match = (len(failed_filters) == 0)

        # -------------------------------
        # Primary semantic gate (floor)
        # -------------------------------
        primary_gate_pass = True
        if is_paragraph_query:
            # Paragraph queries often score lower semantically; only reject
            # very weak ones if we didn't find literal/near-literal evidence.
            if semantic_score < PARAGRAPH_FLOOR and not (literal_hit or near_literal_hit):
                primary_gate_pass = False
        else:
            if is_prompt_role_like and semantic_score < SEMANTIC_FLOOR_ROLE_LIKE:
                primary_gate_pass = False
            if (not is_prompt_role_like) and semantic_score < SEMANTIC_FLOOR_GENERAL:
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
            "experience_display": experience_display,      # added
            "experience_rounded": experience_rounded,      # added
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
            # NEW: surface back what we parsed so UI can reflect if needed
            "schoolsRequired": schools_required,
            "degreesRequired": degrees_required,
            "mustHavePhrases": must_have_phrases,
            "excludePhrases": exclude_phrases,
            "literalHit": literal_hit,
            "nearLiteralHit": near_literal_hit,
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
        if schools_required and "education_schools" not in failed_filters:
            item["strengths"].append("Matches required school(s)")
        if degrees_required and "education_degrees" not in failed_filters:
            item["strengths"].append("Matches required degree(s)")
        if must_have_phrases and "phrases_required" not in failed_filters:
            item["strengths"].append("Contains required phrase(s)")

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


# =========================
# Internal parsers (kept)
# =========================
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


# -----------------------------
# NEW: Education & phrase parsers
# -----------------------------
_DEGREE_TOKENS = [
    "llb", "jd", "llm",
    "bcs", "bs", "b.sc", "bsc", "bba", "b.com", "bcom", "ba", "b.e", "be", "btech", "b.tech",
    "ms", "msc", "m.sc", "mcs", "mba", "m.com", "mcom", "ma", "me", "m.e", "mtech", "m.tech",
    "phd", "dphil"
]

_SCHOOL_HINTS = [
    "university", "college", "institute", "school", "polytechnic", "academy"
]

def _parse_education_requirements(prompt: str) -> Tuple[List[str], List[str]]:
    """
    Extract desired schools & degrees from the *raw* prompt.
    Conservative heuristics to avoid breaking existing logic.
    Returns (schools_required, degrees_required) both lower-cased.
    """
    p = " " + prompt.lower() + " "
    schools: List[str] = []
    degrees: List[str] = []

    # Degrees: any listed degree token mentioned explicitly
    for deg in _DEGREE_TOKENS:
        if re.search(rf"\b{re.escape(deg)}\b", p):
            degrees.append(deg)

    # Schools: simple patterns like "from XYZ", "graduate(d) from XYZ", "alumni of XYZ"
    # Capture up to 6 words after the verb; stop at punctuation.
    school_patterns = [
        r"(?:from|graduate(?:d)?\s+from|alumni\s+of|passed\s+from)\s+([a-zA-Z][a-zA-Z&.\-'\s]{2,80})",
        r"(?:at)\s+([a-zA-Z][a-zA-Z&.\-'\s]{2,80})\s+(?:university|college|institute)"
    ]
    for pat in school_patterns:
        for m in re.finditer(pat, p):
            cand = m.group(1).strip()
            cand = re.sub(r"[,.;].*$", "", cand).strip()
            # keep if it looks like a school name or contains a hint
            if any(h in cand for h in _SCHOOL_HINTS) or len(cand.split()) >= 2:
                schools.append(cand)

    # Also hoover quoted school names like "RMIT", "LUMS"
    for m in re.finditer(r"['\"]([^'\"]{2,80})['\"]", p):
        q = m.group(1).strip()
        if len(q) >= 2 and (any(h in q for h in _SCHOOL_HINTS) or len(q) >= 3):
            schools.append(q)

    # dedupe & lower
    schools = list(dict.fromkeys([s.lower() for s in schools if s]))
    degrees = list(dict.fromkeys([d.lower() for d in degrees if d]))
    return schools, degrees


def _parse_phrase_requirements(prompt: str) -> Tuple[List[str], List[str]]:
    """
    Extract must-have phrases & exclude phrases.
    - Any quoted string becomes a candidate must-have phrase.
    - Phrases after words like 'include/exclude phrase', 'must include', etc.
    """
    p = prompt
    must: List[str] = []
    excl: List[str] = []

    # 1) Quoted phrases → must
    for m in re.finditer(r"['\"]([^'\"]{2,200})['\"]", p):
        must.append(m.group(1).strip())

    # 2) "must include ...", "should include ...", "include phrase ..."
    inc_patterns = [
        r"(?:must|should|need to|needs to|required to)\s+(?:include|contain)\s+([^.;\n]+)",
        r"(?:include|contains?)\s+phrase\s+([^.;\n]+)",
    ]
    for pat in inc_patterns:
        mm = re.search(pat, p, flags=re.IGNORECASE)
        if mm:
            span = mm.group(1)
            parts = re.split(r";|,|\band\b|\bor\b|\/", span)
            for part in parts:
                t = part.strip().strip('"').strip("'")
                if len(t) >= 2:
                    must.append(t)

    # 3) "exclude ..." / "must not include ..."
    exc_patterns = [
        r"(?:exclude|without)\s+([^.;\n]+)",
        r"(?:must|should)\s+not\s+(?:include|contain)\s+([^.;\n]+)"
    ]
    for pat in exc_patterns:
        mm = re.search(pat, p, flags=re.IGNORECASE)
        if mm:
            span = mm.group(1)
            parts = re.split(r";|,|\band\b|\bor\b|\/", span)
            for part in parts:
                t = part.strip().strip('"').strip("'")
                if len(t) >= 2:
                    excl.append(t)

    # dedupe & clean
    def _norm_list(xs: List[str]) -> List[str]:
        seen = {}
        out = []
        for x in xs:
            x2 = re.sub(r"\s+", " ", x).strip()
            xl = x2.lower()
            if xl and xl not in seen:
                seen[xl] = True
                out.append(x2)
        return out

    return _norm_list(must), _norm_list(excl)


def _role_is_related(predicted_role: str, requested_role: str) -> bool:
    """Semantic proximity using sentence embeddings; threshold tuned via config."""
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
        return sim >= ROLE_SIM_RELATED  # related roles
    except Exception:
        return False


def _role_relatedness_score(predicted_role: str, requested_role: Optional[str]) -> float:
    """
    Return a role match score in [0,1].
    - 1.0 exact/same family
    - ~0.95 related (embedding similarity >= ROLE_SIM_HIGH)
    - ~0.8 related (embedding similarity >= ROLE_SIM_RELATED)
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
        if sim >= ROLE_SIM_HIGH:
            return 0.95
        if sim >= ROLE_SIM_RELATED:
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
    - If projects are strictly required by the prompt: 1.0 if present else 0.0
    - Otherwise: mild preference for having projects (0.6 vs 0.4)
    """
    has_proj = _has_projects(projects_field, projects_text)
    if projects_required:
        return 1.0 if has_proj else 0.0
    return 0.6 if has_proj else 0.4
