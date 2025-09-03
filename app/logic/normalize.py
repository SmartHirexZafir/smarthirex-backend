# app/logic/normalize.py
from __future__ import annotations
import re
import os
import json
import unicodedata
from typing import Dict, Any, List, Optional, Set, Tuple

# ------------------------------
# Public API (import these)
# ------------------------------
__all__ = [
    # prompt + keyword utilities
    "normalize_prompt",
    "extract_min_years",
    "extract_year_bounds",   # (min, max)
    "tokenize_keywords",
    "expand_keywords",       # centralized keyword expander
    # role token helper kept for backward compatibility
    "normalize_role_word",
    # normalized-field helpers for ingest/search
    "norm_text",
    "normalize_role",
    "normalize_tokens",
    "to_years_of_experience",
    # additive helpers for parsing (safe; optional to use)
    "normalize_degree",
    "make_search_blob",
    "normalize_location",
    # constants / data (safe to import)
    "SYNONYMS",
    "STOPWORDS",
    "PLURAL_SINGULAR",
    "DEGREE_ALIASES",
    "SECTION_ALIASES",
    "ROLE_KEYWORDS",
    "CATEGORY_KEYWORDS",
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
    "react.js": "react",
    "nodejs": "node",
    "node.js": "node",
    "nextjs": "next",
    "next.js": "next",
    "nuxtjs": "nuxt",
    "nuxt.js": "nuxt",
    "py": "python",
    "tf": "tensorflow",
    "sklearn": "scikit",
    "sk-learn": "scikit",
    "k8s": "kubernetes",
    "gke": "kubernetes",
    "eks": "kubernetes",
    "aks": "kubernetes",
    "sass": "scss",
    "tailwindcss": "tailwind",
    "power bi": "powerbi",
    "power-bi": "powerbi",
    "ms-excel": "excel",
    "msexcel": "excel",
    "msword": "word",
    "ppt": "powerpoint",
    "xd": "adobe xd",
    "ps": "photoshop",
    "tensorflow.js": "tensorflow",
    "c#": "csharp",
    "c sharp": "csharp",
    "c++": "cpp",
    "c plus plus": "cpp",
    ".net": "dotnet",
    "dot net": "dotnet",
    "dotnet": "dotnet",

    # data & ml
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "mlops": "ml ops",

    # clouds
    "gcloud": "gcp",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "amazon web services": "aws",
    "ms azure": "azure",
    "microsoft azure": "azure",

    # roles / areas
    "fe": "frontend",
    "front end": "frontend",
    "front-end": "frontend",
    "be": "backend",
    "back end": "backend",
    "back-end": "backend",
    "fullstack": "full-stack",
    "data science": "data scientist",
    "data engineering": "data engineer",
    "software developer": "software engineer",
    "ui/ux": "ui ux",
    "uix": "ui ux",
    "hr": "human resources",
    "pr": "public relations",
    "qa": "qa engineer",

    # HR/Admin
    "hrm": "hr manager",
    "hrd": "hr director",

    # misc suites/platforms
    "msexcel": "excel",
    "ms word": "word",
    "google ads": "google ads",

    # operator/constraint phrases → canonical symbols (broad coverage for filters)
    "less than or equal to": "<=",
    "less than equal to": "<=",
    "at most": "<=",
    "no more than": "<=",
    "up to": "<=",
    "greater than or equal to": ">=",
    "at least": ">=",
    "minimum": ">=",
    "min": ">=",
    "more than": ">",
    "over": ">",
    "above": ">",
    "less than": "<",
    "under": "<",
    "below": "<",
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

# ----------------------- Optional externalized config (no hardcoding) --------
# Allow project-specific overrides via resource files; fallback to above.

_CUSTOM_STOPWORDS_PATHS: List[str] = [
    os.path.join("app", "resources", "stopwords_custom.txt"),
    os.path.join("resources", "stopwords_custom.txt"),
    "stopwords_custom.txt",
]

_CUSTOM_PLURALS_PATHS: List[str] = [
    os.path.join("app", "resources", "plurals_custom.json"),
    os.path.join("resources", "plurals_custom.json"),
    "plurals_custom.json",
]


def _merge_custom_stopwords() -> None:
    for path in _CUSTOM_STOPWORDS_PATHS:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if not w or w.startswith("#"):
                        continue
                    STOPWORDS.add(w)
        except Exception:
            continue


def _merge_custom_plurals() -> None:
    for path in _CUSTOM_PLURALS_PATHS:
        try:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str):
                        kk = k.strip().lower()
                        vv = v.strip().lower()
                        if kk and vv:
                            PLURAL_SINGULAR[kk] = vv
        except Exception:
            continue


_merge_custom_stopwords()
_merge_custom_plurals()

# ----------------------- number-word helpers (broader coverage) ---------------
# Support words like "four", "twenty-five", etc., used in prompts
_NUM_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
    # small fractions often used in experience text/prompts
    "half": 0.5,
}


def _word_to_number(tok: str) -> Optional[float]:
    """
    Convert a word or hyphenated word number to a float.
    Examples: 'four'->4, 'twenty-five'->25, 'one-and-a-half'->1.5 (best-effort)
    """
    t = tok.strip().lower()
    if t.isdigit():
        return float(t)
    if t in _NUM_WORDS:
        return float(_NUM_WORDS[t])
    # simple fraction like 1/2
    if re.match(r"^\d+/\d+$", t):
        try:
            num, den = t.split("/")
            return float(num) / float(den)
        except Exception:
            return None
    # "one and a half", "two and half"
    m = re.match(r"^([a-z\-]+)\s+and\s+(?:a\s+)?half$", t)
    if m:
        base = _word_to_number(m.group(1))
        if base is not None:
            return base + 0.5
    if "-" in t:
        parts = t.split("-")
        if all(p in _NUM_WORDS for p in parts):
            total = 0.0
            for p in parts:
                v = _NUM_WORDS[p]
                if v == 100 and total > 0:
                    total *= v
                else:
                    total += v
            return float(total)
    return None

# -----------------------------------------------------------------------------
# Core normalizers used by ingest + search
# -----------------------------------------------------------------------------

def _normalize_unicode_operators(s: str) -> str:
    """
    Map common unicode comparators to ASCII operators for consistent parsing.
    """
    return (
        s.replace("≤", "<=")
         .replace("≦", "<=")
         .replace("⩽", "<=")
         .replace("≥", ">=")
         .replace("≧", ">=")
         .replace("⩾", ">=")
         .replace("＜", "<")
         .replace("＞", ">")
    )


def _strip_diacritics(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        return s


def _basic_clean(s: str) -> str:
    """Lowercase + trim spaces (with diacritic + operator normalization)."""
    s = (s or "")
    s = _strip_diacritics(s)
    s = _normalize_unicode_operators(s)
    return re.sub(r"\s+", " ", s.lower()).strip()


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
        # try number words (e.g., "five years")
        parts = s.split()
        for p in parts:
            n = _word_to_number(p)
            if n is not None:
                return int(n)
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
    (r"\bnuxt\.?js\b", "nuxt"),
    (r"\bc\s*#\b", "csharp"),
    (r"\bc\s*\+\+\b", "cpp"),
    (r"\b\.?dot\s*net\b", "dotnet"),

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
    (r"\bgoogle\s*ads\b", "google ads"),

    # infra
    (r"\bgoogle\s*cloud\s*platform\b", "gcp"),
    (r"\bamazon\s*web\s*services\b", "aws"),
    (r"\bmicro\s*services\b", "microservices"),

    # ui/ux
    (r"\bui/?ux\b", "ui ux"),

    # operator/constraint phrases (safety even if SYNONYMS handles them)
    (r"\bless\s+than\s+or\s+equal\s+to\b", "<="),
    (r"\bless\s+than\s+equal\s+to\b", "<="),
    (r"\bat\s+most\b", "<="),
    (r"\bno\s+more\s+than\b", "<="),
    (r"\bup\s+to\b", "<="),
    (r"\bgreater\s+than\s+or\s+equal\s+to\b", ">="),
    (r"\bat\s+least\b", ">="),
    (r"\bminimum\b", ">="),
    (r"\bmin\b", ">="),
    (r"\bmore\s+than\b", ">"),
    (r"\bover\b", ">"),
    (r"\babove\b", ">"),
    (r"\bless\s+than\b", "<"),
    (r"\bunder\b", "<"),
    (r"\bbelow\b", "<"),
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
    "public relations",
    "financial modeling",
    "microservices",
    "business analyst",
    "network security",
    "quality assurance",
    "operations manager",
    "web designing",
]


def _apply_phrase_normalizers(s: str) -> str:
    s = _normalize_unicode_operators(s)
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
    # allow + for "3+" patterns and keep comparator symbols <> =
    s = re.sub(r"[^a-z0-9+_<>= _\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # First, extract the marked phrases
    tokens: List[str] = []
    idx = 0
    while idx < len(s):
        m = re.search(r"__PHRASE__([a-z0-9 \-]+?)__PHRASE__", s[idx:])
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


def _numberify_variants(tok: str) -> List[str]:
    """
    Return a list containing tok and (if applicable) its numeric form.
    Keeps order and uniqueness handled by caller.
    Examples:
      'four' -> ['four', '4']
      'five+' -> ['five+', '5+']
      'twenty-five' -> ['twenty-five', '25']
    """
    variants = [tok]
    plus = tok.endswith("+")
    core = tok[:-1] if plus else tok
    num = _word_to_number(core)
    if num is not None:
        numeric = f"{int(num) if float(num).is_integer() else num}"
        if plus:
            numeric += "+"
        if numeric not in variants:
            variants.append(numeric)
    return variants


def tokenize_keywords(s: str) -> List[str]:
    """
    Tokenize into keywords: keep [a-z0-9 + - < > =], drop stopwords, keep distinct order.
    Preserves important bigrams/trigrams (e.g., 'machine learning', 'data science', 'ui ux').
    Also preserves tokens like '3+' to help min years detection upstream.
    Additionally, includes numeric variants for number-words (e.g., 'four' -> '4').
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
        # if phrase (contains space) always keep as-is
        if " " in t:
            if t not in seen:
                seen.add(t)
                out.append(t)
            continue
        # single word -> filter on stopwords
        if t in STOPWORDS:
            # but if it is a number word we still want the numeric variant
            variants = _numberify_variants(t)
            for v in variants:
                if v not in seen and v not in STOPWORDS:
                    seen.add(v)
                    out.append(v)
            continue
        # include token and (if applicable) its numeric/comparator variant
        for v in _numberify_variants(t):
            if v not in seen:
                seen.add(v)
                out.append(v)
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
      - number-words: 'four years', 'at least five years', 'between three and five years'
    Returns float or None.
    """
    if not prompt_like:
        return None

    s = _basic_clean(prompt_like)

    # normalize operator phrases to symbols (kept consistent with SYNONYMS / phrase normalizers)
    s = _apply_phrase_normalizers(s)
    s = _apply_plural_and_synonyms(s)

    def _to_num(token: str) -> Optional[float]:
        # token may be digits or number words (hyphenated)
        token = token.strip().lower()
        if re.match(r"^\d+(?:\.\d+)?$", token):
            try:
                return float(token)
            except Exception:
                return None
        return _word_to_number(token)

    # Ranges → return the lower bound (supports words or digits)
    m = re.search(r"\bbetween\s+([a-z0-9\-\.]+)\s+(?:and|to)\s+([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        a = _to_num(m.group(1))
        b = _to_num(m.group(2))
        if a is not None and b is not None:
            return float(min(a, b))

    m = re.search(r"\b([a-z0-9\-\.]+)\s*(?:-|–|—|to)\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        a = _to_num(m.group(1))
        b = _to_num(m.group(2))
        if a is not None and b is not None:
            return float(min(a, b))

    # Symbols & phrasing that imply a minimum
    m = re.search(r"\b>=\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return float(n)

    m = re.search(r"\b>=\s*([a-z0-9\-\.]+)\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return float(n)

    # 'at least' / 'minimum' / 'min' (already normalized to '>=' above, but keep extra safety)
    m = re.search(r"\b(at\s*least|min(?:imum)?)\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(2))
        if n is not None:
            return float(n)

    # 'more than / over / above' → slight nudge above number
    m = re.search(r"\b(>|more than|over|above)\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(2))
        if n is not None:
            return float(n) + 0.01

    # "X+ years" or "X+"
    m = re.search(r"\b([a-z0-9\-\.]+)\s*\+\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return float(n)

    # Plain "X years" or just "X" near 'years' → treat as minimum X
    m = re.search(r"\b([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return float(n)

    # If we explicitly see a maximum-only constraint (<= or <), do NOT return a min
    if re.search(r"\b(?:<=|<)\s*[a-z0-9\-\.]+\s*(?:years?|yrs?)?\b", s):
        return None

    # No usable minimum found
    return None


# Extract both min and max years from a prompt
def extract_year_bounds(prompt_like: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract (min_years, max_years) from a prompt.
    Supports:
      - '>= 3 years', 'at least three years'   → min=3
      - '<= 5 years', 'up to five years'       → max=5
      - '< 5 years'                             → max=5  (treated as inclusive bound downstream)
      - '> 3 years'                             → min=3  (treated as inclusive bound downstream)
      - 'between three and five years'          → min=3, max=5
      - '3-5 years', '3 to 5 yrs'               → min=3, max=5
      - plain '4 years'                         → min=4
      - number-words & hyphenated numbers supported
    Returns (min_years, max_years) with each possibly None.
    """
    if not prompt_like:
        return (None, None)

    s = _basic_clean(prompt_like)
    s = _apply_phrase_normalizers(s)
    s = _apply_plural_and_synonyms(s)

    def _to_num(token: str) -> Optional[float]:
        token = token.strip().lower()
        if re.match(r"^\d+(?:\.\d+)?$", token):
            try:
                return float(token)
            except Exception:
                return None
        return _word_to_number(token)

    # between X and Y years
    m = re.search(r"\bbetween\s+([a-z0-9\-\.]+)\s+(?:and|to)\s+([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        a = _to_num(m.group(1))
        b = _to_num(m.group(2))
        if a is not None and b is not None:
            lo, hi = sorted([a, b])
            return (float(lo), float(hi))

    # X-Y years or X to Y years
    m = re.search(r"\b([a-z0-9\-\.]+)\s*(?:-|–|—|to)\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        a = _to_num(m.group(1))
        b = _to_num(m.group(2))
        if a is not None and b is not None:
            lo, hi = sorted([a, b])
            return (float(lo), float(hi))

    # <= N years
    m = re.search(r"\b<=\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (None, float(n))

    # >= N years
    m = re.search(r"\b>=\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (float(n), None)

    # Strict < / >
    m = re.search(r"\b<\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (None, float(n))
    m = re.search(r"\b>\s*([a-z0-9\-\.]+)\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (float(n), None)

    # X+ years
    m = re.search(r"\b([a-z0-9\-\.]+)\s*\+\s*(?:years?|yrs?)?\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (float(n), None)

    # plain "X years" -> minimum
    m = re.search(r"\b([a-z0-9\-\.]+)\s*(?:years?|yrs?)\b", s)
    if m:
        n = _to_num(m.group(1))
        if n is not None:
            return (float(n), None)

    return (None, None)


# Centralized keyword expansion (numbers & synonyms in both directions)
def expand_keywords(keywords: List[str]) -> List[str]:
    """
    Expand a list of keywords by:
      - mapping number-words to digits (e.g., 'four' → '4')
      - adding synonyms (via SYNONYMS)
      - inverting synonyms (if a token matches a synonym value, add the key)
    Returns a de-duplicated, lowercased list.
    """
    syn = {k.strip().lower(): v.strip().lower() for k, v in SYNONYMS.items() if k and v}
    out: Set[str] = set()
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
            out.add(str(int(n) if float(n).is_integer() else n))
        # synonyms forward
        if k_l in syn:
            out.add(syn[k_l])
        # inverse: if token equals a synonym value, include keys that map to it
        for root, val in syn.items():
            if k_l == val:
                out.add(root)
    return sorted(out)


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

    # Phrase fixes first (react-native, ms excel, ui/ux, operator phrases, etc.)
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

# -----------------------------------------------------------------------------
# Additive helpers for CV parsing quality (Req #8 supportive utilities)
# -----------------------------------------------------------------------------

# Canonical degree aliases (safe, additive)
DEGREE_ALIASES: Dict[str, str] = {
    # bachelors
    "b.sc": "bsc", "bsc": "bsc", "b s c": "bsc", "bachelor of science": "bsc",
    "b.tech": "btech", "btech": "btech", "bachelor of technology": "btech",
    "b.e": "be", "be": "be", "bachelor of engineering": "be",
    "ba": "ba", "b.a": "ba", "bachelor of arts": "ba",
    "bca": "bca", "b.com": "bcom", "bcom": "bcom",
    # masters
    "m.sc": "msc", "msc": "msc", "master of science": "msc",
    "m.tech": "mtech", "mtech": "mtech", "master of technology": "mtech",
    "m.e": "me", "me": "me", "master of engineering": "me",
    "ma": "ma", "m.a": "ma", "master of arts": "ma",
    "mca": "mca", "mba": "mba", "pgdm": "pgdm",
    # doctorate / law
    "ph.d": "phd", "phd": "phd", "doctor of philosophy": "phd",
    "ll.b": "llb", "llb": "llb", "ll.m": "llm", "llm": "llm", "juris doctor": "jd", "jd": "jd",
}

def normalize_degree(s: Optional[str]) -> Optional[str]:
    """
    Normalize a degree string to a canonical code (e.g., "B.Sc" → "bsc").
    Non-destructive; returns None for empty input.
    """
    if not s:
        return None
    t = _basic_clean(s)
    t = t.replace(".", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # try direct alias map
    if t in DEGREE_ALIASES:
        return DEGREE_ALIASES[t]
    # try collapsing spaces
    t2 = t.replace(" ", "")
    if t2 in DEGREE_ALIASES:
        return DEGREE_ALIASES[t2]
    # loose contains for common long forms
    for k, v in DEGREE_ALIASES.items():
        if k in t:
            return v
    return t or None


def normalize_location(s: Optional[str]) -> Optional[str]:
    """
    Very light location normalizer: lowercase, collapse spaces and commas.
    Keeps city, country order intact; safe for indexing.
    """
    if not s:
        return None
    t = _strip_diacritics(s).strip()
    # unify comma spacing
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def make_search_blob(parts: List[str], truncate_chars: Optional[int] = None) -> str:
    """
    Build a normalized search blob from multiple parts (name, role, skills, projects, summary, raw).
    Mirrors the light normalization used in resume parsing; safe, non-destructive.
    """
    joined = " ".join([p for p in (parts or []) if isinstance(p, str) and p])
    if truncate_chars is None:
        try:
            truncate_chars = int(os.getenv("SEARCH_BLOB_TRUNCATE_CHARS", "4000"))
        except Exception:
            truncate_chars = 4000
    blob = joined[:max(0, truncate_chars)]
    # lowercase + keep alphanumerics and a few separators
    blob = blob.lower()
    blob = re.sub(r"http\S+|www\S+|https\S+", " ", blob)
    blob = re.sub(r"\S+@\S+", " ", blob)
    blob = re.sub(r"[^a-z0-9\s+/.,_-]", " ", blob)
    blob = re.sub(r"\s+", " ", blob).strip()
    return blob

# ----------------------- Section aliases (helpful for parsing) ----------------
SECTION_ALIASES: Dict[str, List[str]] = {
    "summary": ["summary", "objective", "profile", "professional summary"],
    "skills": ["skills", "technical skills", "core skills", "skills summary"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history"],
    "projects": ["projects", "project highlights", "key projects", "academic projects"],
    "education": ["education", "academic qualifications", "academics", "qualifications"],
    "certifications": ["certifications", "licenses", "certificates"],
}

# ----------------------- Role / Category keyword dictionaries -----------------
# Optional: load from JSON if available; otherwise keep empty (parser may have fallback)
_ROLE_CAT_JSON_CANDIDATES = [
    os.path.join("app", "resources", "role_category_keywords.json"),
    os.path.join("resources", "role_category_keywords.json"),
    "role_category_keywords.json",
]

ROLE_KEYWORDS: Dict[str, List[str]] = {}
CATEGORY_KEYWORDS: Dict[str, List[str]] = {}

def _load_role_category_keywords() -> None:
    global ROLE_KEYWORDS, CATEGORY_KEYWORDS
    for p in _ROLE_CAT_JSON_CANDIDATES:
        try:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            rk = data.get("role_keywords") or data.get("roles") or {}
            ck = data.get("category_keywords") or data.get("categories") or {}
            if isinstance(rk, dict) and rk:
                ROLE_KEYWORDS = {str(k): [str(x).lower() for x in (v or []) if isinstance(x, str)] for k, v in rk.items()}
            if isinstance(ck, dict) and ck:
                CATEGORY_KEYWORDS = {str(k): [str(x).lower() for x in (v or []) if isinstance(x, str)] for k, v in ck.items()}
            break
        except Exception:
            # ignore and try next path
            continue

_load_role_category_keywords()
