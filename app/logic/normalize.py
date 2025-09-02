# app/logic/normalize.py
from __future__ import annotations
import re
import os
import json
from typing import Dict, Any, List, Optional, Set

# ------------------------------
# Public API (import these)
# ------------------------------
__all__ = [
    # prompt + keyword utilities
    "normalize_prompt",
    "extract_min_years",
    "tokenize_keywords",
    # role token helper kept for backward compatibility
    "normalize_role_word",
    # new normalized-field helpers for ingest/search
    "norm_text",
    "normalize_role",
    "normalize_tokens",
    "to_years_of_experience",
    # constants
    "SYNONYMS",
    "STOPWORDS",
]

# ------------------------------
# Synonyms / Stopwords / Helpers
# ------------------------------

# Common shortforms & variants -> canonical terms
# (safe, additive; tech + business + design + HR)
SYNONYMS: Dict[str, str] = {
    # technologies
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "nodejs": "node",
    "nextjs": "next",
    "next.js": "next",
    "py": "python",
    "tf": "tensorflow",
    "sklearn": "scikit",
    "sk-learn": "scikit",
    "k8s": "kubernetes",
    "gke": "kubernetes",
    "eks": "kubernetes",
    "aks": "kubernetes",
    "sass": "scss",
    "power bi": "powerbi",
    "power-bi": "powerbi",

    # data & ml
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "mlops": "ml ops",
    "tensorflow.js": "tensorflow",

    # clouds
    "gcloud": "gcp",
    "google cloud": "gcp",
    "amazon web services": "aws",
    "ms azure": "azure",

    # roles / areas
    "fe": "frontend",
    "be": "backend",
    "fullstack": "full-stack",
    "data science": "data scientist",
    "data engineering": "data engineer",
    "software developer": "software engineer",
    "ui/ux": "ui ux",

    # HR/Admin
    "hrm": "hr manager",
    "hrd": "hr director",

    # misc tools/platforms
    "ppt": "powerpoint",
    "msword": "word",
    "ms-excel": "excel",
    "msexcel": "excel",
    "xd": "adobe xd",
    "ps": "photoshop",
    "tailwindcss": "tailwind",
}

# ---- Optional custom synonyms merge (non-breaking) --------------------------
_CUSTOM_SYNONYMS_PATHS: List[str] = [
    os.path.join("app", "resources", "synonyms_custom.json"),
    os.path.join("resources", "synonyms_custom.json"),
    "synonyms_custom.json",
]


def _merge_custom_synonyms() -> None:
    """
    Load custom synonyms and merge into SYNONYMS.
    - Keys/values are lowercased and stripped.
    - Invalid entries are ignored.
    - Custom entries override defaults.
    Silently no-ops if file(s) are missing or invalid.
    """
    for path in _CUSTOM_SYNONYMS_PATHS:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            cleaned: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    kk = k.strip().lower()
                    vv = v.strip().lower()
                    if kk and vv:
                        cleaned[kk] = vv
            if cleaned:
                SYNONYMS.update(cleaned)
        except Exception:
            # Optional file — ignore any errors to preserve old behavior
            continue


# Attempt merge at import time (safe if file is absent)
_merge_custom_synonyms()

# -----------------------------------------------------------------------------


# Words we ignore while generating keywords
STOPWORDS: Set[str] = {
    # generic
    "show", "find", "me", "with", "and", "or", "in", "of", "for", "to",
    "on", "by", "at", "from", "that", "who", "which",
    "candidates", "candidate", "experience", "years", "based",
    "a", "an", "the", "please", "pls",

    # role terms we usually don’t want to force as keywords
    "developer", "developers", "engineer", "engineers",
    "scientist", "scientists", "designer", "designers",
    "manager", "managers", "analyst", "analysts",

    # filler/action words
    "looking", "seek", "seeking", "needed", "need", "needs",
    "require", "required", "requirements", "must", "should", "prefer",
    "preferred", "bonus", "nice", "solid", "strong", "good",
    "knowledge", "skills", "skill", "tools", "background", "exposure",
    "ability", "abilities", "proficiency", "proficient",
    "hands-on", "hands", "plus",

    # process verbs we don't want as keywords
    "building", "build", "develop", "developing", "deploy", "deploying",
    "maintain", "maintaining", "manage", "managing",
}

# Known plural forms we collapse to singular (minimal, safe)
PLURAL_SINGULAR: Dict[str, str] = {
    "developers": "developer",
    "engineers": "engineer",
    "scientists": "scientist",
    "designers": "designer",
    "managers": "manager",
    "analysts": "analyst",
    "architects": "architect",
    "candidates": "candidate",
}

# -----------------------------------------------------------------------------
# Core normalizers used by ingest + search (added per our plan)
# -----------------------------------------------------------------------------


def _basic_clean(s: str) -> str:
    """Lowercase + trim spaces."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def norm_text(s: Optional[str]) -> Optional[str]:
    """Minimal, safe normalizer for free text fields (keeps None)."""
    if s is None:
        return None
    out = _basic_clean(s)
    return out if out else None


def normalize_role(s: Optional[str]) -> Optional[str]:
    """
    Normalize a role/title into a canonical, search-friendly form.
    Examples:
      " Full Stack Developer " -> "full-stack developer" (via phrase + synonyms)
      "BE Engineer"            -> "backend engineer"
    """
    s = norm_text(s)
    if not s:
        return None
    # stabilize common phrases first
    s = _apply_phrase_normalizers(s)
    # unify "full stack" → "full-stack", etc. (already covered but reinforced)
    s = re.sub(r"\bfull\s*stack\b", "full-stack", s)
    # token-level cleanup (plurals + synonyms)
    s = _apply_plural_and_synonyms(s)
    return s


def normalize_tokens(items: Optional[List[str]]) -> List[str]:
    """
    Normalize a list of tokens (skills/projects/etc.) into a de-duplicated,
    lowercased list with synonyms applied.
    """
    if not items:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if not isinstance(x, str):
            continue
        t = norm_text(x)
        if not t:
            continue
        # phrase fixes then token-level maps
        t = _apply_phrase_normalizers(t)
        t = _apply_plural_and_synonyms(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


_YOE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def to_years_of_experience(v: Optional[str | int | float]) -> Optional[int]:
    """
    Parse a years-of-experience value into an int (floor).
    Accepts numeric values or strings like "3", "3.5", "3 years", "5+ yrs".
    Returns None if no numeric content found.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return int(float(v))
        except Exception:
            return None
    s = _basic_clean(str(v))
    m = _YOE_NUM_RE.search(s)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None


def normalize_role_word(w: str) -> str:
    """Collapse common plurals and apply synonyms (role-like tokens)."""
    w = _basic_clean(w)
    w = PLURAL_SINGULAR.get(w, w)
    # apply synonyms (only if exact token match)
    if w in SYNONYMS:
        w = SYNONYMS[w]
    return w


# Phrases we want to normalize/capture as single tokens
# (run before token-level synonyms so multiword forms are stabilized)
_PHRASE_NORMALIZERS: List[tuple[str, str]] = [
    # office / suites
    (r"\bms[\s\-]?excel\b", "excel"),
    (r"\bms[\s\-]?word\b", "word"),
    (r"\bpower\s*bi\b", "powerbi"),
    (r"\bpower\s*point\b", "powerpoint"),

    # front/back/full
    (r"\bfront[\- ]?end\b", "frontend"),
    (r"\bback[\- ]?end\b", "backend"),
    (r"\bfull[\- ]?stack\b", "full-stack"),

    # web & libs
    (r"\breact[\s\-]?native\b", "react native"),
    (r"\bnode\.?js\b", "node"),
    (r"\bnext\.?js\b", "next"),

    # adobe
    (r"\badobe\s*xd\b", "adobe xd"),
    (r"\b(adobe )?illustrator\b", "adobe illustrator"),
    (r"\b(adobe )?photoshop\b", "photoshop"),

    # ml areas
    (r"\bmachine\s*learning\b", "machine learning"),
    (r"\bdeep\s*learning\b", "deep learning"),
    (r"\bnatural\s*language\s*processing\b", "natural language processing"),
    (r"\bcomputer\s*vision\b", "computer vision"),
    (r"\bdata\s*science\b", "data science"),
    (r"\bdata\s*engineering\b", "data engineering"),
    (r"\bml[\s\-]?ops\b", "ml ops"),

    # analytics / viz
    (r"\btableau\b", "tableau"),
    (r"\bpower\s*bi\b", "powerbi"),

    # infra
    (r"\bgoogle\s*cloud\s*platform\b", "gcp"),
    (r"\bamazon\s*web\s*services\b", "aws"),
    (r"\bmicro\s*services\b", "microservices"),

    # ui/ux
    (r"\bui/?ux\b", "ui ux"),
]

# Phrases to keep as a single keyword if present (bigram/trigram keepers)
# NOTE: keep in lowercase canonical form
_KEEP_PHRASES: List[str] = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "data science",
    "data engineer",
    "data scientist",
    "data analyst",
    "data engineering",
    "ml ops",
    "react native",
    "ui ux",
    "powerbi",
    "powerpoint",
    "google ads",
    "project management",
    "supply chain",
    "human resources",
    "financial modeling",
    "microservices",
]


def _apply_phrase_normalizers(s: str) -> str:
    for pat, repl in _PHRASE_NORMALIZERS:
        s = re.sub(pat, repl, s)
    return s


def _apply_plural_and_synonyms(s: str) -> str:
    """Apply plural→singular and synonyms across a sentence."""
    # 1) plural→singular (word-boundary safe)
    for pl, sg in PLURAL_SINGULAR.items():
        s = re.sub(rf"\b{re.escape(pl)}\b", sg, s)
    # 2) token-level synonyms (defaults + optional custom merged above)
    for k, v in SYNONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _inject_keep_phrase_markers(s: str) -> str:
    """
    Wrap keep-phrases with markers so they stay as one token during tokenization.
    Example: 'machine learning' -> '__PHRASE__machine learning__PHRASE__'
    """
    for phrase in sorted(_KEEP_PHRASES, key=len, reverse=True):
        p = re.escape(phrase)
        s = re.sub(rf"\b{p}\b", f"__PHRASE__{phrase}__PHRASE__", s)
    return s


def _split_to_tokens_keep_phrases(s: str) -> List[str]:
    """
    Convert sentence to tokens while preserving keep-phrases as a single token.
    """
    # allow + for "3+" patterns
    s = re.sub(r"[^a-z0-9+ _]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # First, extract the marked phrases
    tokens: List[str] = []
    idx = 0
    while idx < len(s):
        m = re.search(r"__PHRASE__([a-z0-9 ]+?)__PHRASE__", s[idx:])
        if not m:
            rest = s[idx:].strip()
            if rest:
                tokens.extend(rest.split())
            break
        start = idx + m.start()
        end = idx + m.end()
        # add any plain tokens before the phrase
        before = s[idx:start].strip()
        if before:
            tokens.extend(before.split())
        # add the phrase as one token (keep original spacing inside)
        phrase_token = m.group(1).strip()
        if phrase_token:
            tokens.append(phrase_token)
        idx = end
    return tokens


def tokenize_keywords(s: str) -> List[str]:
    """
    Tokenize into keywords: keep [a-z0-9 +], drop stopwords, keep distinct order.
    Preserves important bigrams/trigrams (e.g., 'machine learning', 'data science', 'ui ux').
    Also preserves tokens like '3+' to help min years detection upstream.
    """
    s = _basic_clean(s)
    s = _inject_keep_phrase_markers(s)
    tokens = _split_to_tokens_keep_phrases(s)

    # remove stopwords (but NEVER drop kept phrases)
    out: List[str] = []
    seen: Set[str] = set()
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # if phrase (contains space) always keep
        if " " in t:
            if t not in seen:
                seen.add(t)
                out.append(t)
            continue
        # single word -> filter on stopwords
        if t in STOPWORDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_min_years(prompt_like: str) -> Optional[float]:
    """
    Parse minimum years if present (lower-bound intent).
    Supports:
      - '3+ years', '3 years', '3+yrs'
      - 'at least 3 years', 'minimum 3 years', 'min 3'
      - 'more than 3 years', 'over 3 years' (nudged to 3.01)
      - ranges: 'between 3 and 5 years', '3-5 years', '3 to 5 yrs' -> returns 3
      - symbols: '>= 3 years'
    Returns float or None.
    """
    if not prompt_like:
        return None

    s = _basic_clean(prompt_like)

    # Ranges → return the lower bound
    m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        try:
            a, _ = float(m.group(1)), float(m.group(2))
            return a
        except Exception:
            pass

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        try:
            a, _ = float(m.group(1)), float(m.group(2))
            return a
        except Exception:
            pass

    # Symbols & phrasing that imply a minimum
    m = re.search(r"\b>=\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        return float(m.group(1))

    m = re.search(r"\b(min(?:imum)?|at least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        return float(m.group(2))

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\b", s)
    if m:
        return float(m.group(1))

    # 'more than / over' → slight nudge above number
    m = re.search(r"\b(more than|over)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        return float(m.group(2)) + 0.01

    # Plain "X years" → treat as minimum X
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", s)
    if m:
        return float(m.group(1))

    # No usable minimum found
    return None


def normalize_prompt(raw: str) -> Dict[str, Any]:
    """
    Normalize prompt text and extract lightweight keywords.
    Returns:
      {
        "normalized_prompt": <str>,
        "keywords": <List[str]>,        # includes important phrases (bigrams)
        "is_role_like": <bool>          # <= 4 words (simple heuristic)
      }
    """
    s = _basic_clean(raw)

    # Phrase fixes first (react-native, ms excel, ui/ux, etc.)
    s = _apply_phrase_normalizers(s)

    # Then plurals/synonyms (token level) — includes optional custom overrides
    s = _apply_plural_and_synonyms(s)

    # Finally, generate keywords (keeps bigrams like "machine learning", "ui ux")
    keywords = tokenize_keywords(s)

    # Heuristic for short "role-only" prompts
    is_role_like = len(s.split()) <= 4

    return {
        "normalized_prompt": s,
        "keywords": keywords,
        "is_role_like": is_role_like,
    }
