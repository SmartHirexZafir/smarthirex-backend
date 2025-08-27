# app/logic/normalize.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Set

# ------------------------------
# Public API (import these)
# ------------------------------
__all__ = [
    "normalize_prompt",
    "extract_min_years",
    "tokenize_keywords",
    "normalize_role_word",
    "SYNONYMS",
    "STOPWORDS",
]

# ------------------------------
# Synonyms / Stopwords / Helpers
# ------------------------------

# Common shortforms & variants -> canonical terms
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

    # roles / areas
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "fe": "frontend",
    "be": "backend",
    "fullstack": "full-stack",

    # misc
    "pg": "postgres",
    "gcp": "google cloud",
}

# Words we ignore while generating keywords
STOPWORDS: Set[str] = {
    "show", "find", "me", "with", "and", "in", "of", "for", "to",
    "candidates", "candidate", "experience", "years", "based",
    "a", "an", "the", "please", "pls",
    # role terms we usually don’t want to force as keywords
    "developer", "developers", "engineer", "engineers",
    "scientist", "scientists", "designer", "designers",
    "manager", "managers",
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


def _basic_clean(s: str) -> str:
    """Lowercase + trim spaces."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def normalize_role_word(w: str) -> str:
    """Collapse common plurals and apply synonyms (role-like tokens)."""
    w = _basic_clean(w)
    w = PLURAL_SINGULAR.get(w, w)
    # apply synonyms (only if exact token match)
    if w in SYNONYMS:
        w = SYNONYMS[w]
    return w


def _apply_plural_and_synonyms(s: str) -> str:
    """Apply plural→singular and synonyms across a sentence."""
    # 1) plural→singular (word-boundary safe)
    for pl, sg in PLURAL_SINGULAR.items():
        s = re.sub(rf"\b{re.escape(pl)}\b", sg, s)

    # 2) synonyms
    for k, v in SYNONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def tokenize_keywords(s: str) -> List[str]:
    """
    Tokenize into keywords: keep [a-z0-9 +], drop stopwords, keep distinct order.
    Also preserves tokens like '3+' to help min years detection upstream.
    """
    s = _basic_clean(s)
    s = re.sub(r"[^a-z0-9+ ]", " ", s)
    tokens = s.split()
    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    # de-dup while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_min_years(prompt_like: str) -> Optional[float]:
    """
    Parse numbers like:
      - '3+ years', '3 years', '3+yrs'
      - 'at least 3 years', 'minimum 3 years'
      - 'more than 3 years', 'over 3 years'
    Returns float or None.
    """
    if not prompt_like:
        return None

    s = _basic_clean(prompt_like)

    patterns = [
        r"(?:minimum|at least)\s*(\d+(?:\.\d+)?)\s*(?:years|yrs?)",
        r"(?:more than|over)\s*(\d+(?:\.\d+)?)\s*(?:years|yrs?)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs?)",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            try:
                val = float(m.group(1))
            except Exception:
                continue
            # tiny nudge for “more than / over”
            if re.search(r"\b(more than|over)\b", s):
                val += 0.01
            return val
    return None


def normalize_prompt(raw: str) -> Dict[str, Any]:
    """
    Normalize prompt text and extract lightweight keywords.
    Returns:
      {
        "normalized_prompt": <str>,
        "keywords": <List[str]>,
        "is_role_like": <bool>   # <= 4 words (simple heuristic)
      }
    """
    s = _basic_clean(raw)
    s = _apply_plural_and_synonyms(s)
    keywords = tokenize_keywords(s)
    is_role_like = len(s.split()) <= 4
    return {
        "normalized_prompt": s,
        "keywords": keywords,
        "is_role_like": is_role_like,
    }
