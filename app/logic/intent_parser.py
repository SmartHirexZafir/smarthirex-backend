# ✅ File: app/logic/intent_parser.py

import re
import os
import json
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------
# Existing logic (unchanged APIs preserved)
# ---------------------------

def detect_usage_help(prompt: str) -> bool:
    """
    Detects if the user is asking for help/about information.
    Only triggers if user is clearly asking 'how to use' the system.
    """
    prompt = prompt.lower()
    help_patterns = [
        r"\bhow to\b",
        r"\bhelp\b",
        r"\bkaise\b",
        r"\buse\b",
        r"\bstart\b",
        r"\bregister\b",
        r"\bsign ?up\b",
        r"\bguide\b"
    ]
    return any(re.search(pattern, prompt) for pattern in help_patterns)

def detect_show_all(prompt: str) -> bool:
    """
    Detects if the user wants to see all resumes, without any filter.
    """
    prompt = prompt.strip().lower()
    return prompt in [
        "show all",
        "list all",
        "show all candidates",
        "list all resumes",
        "display all",
        "get all",
        "sab dikhao",
        "saare candidate",
        # ✅ a few extra phrasings
        "show all cvs",
        "list all cvs",
        "all cvs",
        "all resumes",
        "saare cvs",
        "saari cvs",
    ]

# ---------------------------
# NEW: Optional roles lexicon (non-breaking)
# ---------------------------

_ROLES_LEXICON_PATH = os.path.join("app", "resources", "roles_lexicon.json")
_ROLES_LEXICON: Dict[str, str] = {}  # variant -> canonical

def _norm_space(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s_\-\/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _load_roles_lexicon() -> None:
    """
    Load roles_lexicon.json if present. Expected shape:
      { "variants_to_canonical": { "<variant>": "<canonical>", ... } }
    Silently no-op if file missing/invalid. Custom entries override built-ins if added later.
    """
    global _ROLES_LEXICON
    try:
        with open(_ROLES_LEXICON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        map_ = data.get("variants_to_canonical", {})
        if isinstance(map_, dict):
            cleaned: Dict[str, str] = {}
            for k, v in map_.items():
                if isinstance(k, str) and isinstance(v, str):
                    nk = _norm_space(k)
                    nv = _norm_space(v)
                    if nk and nv:
                        cleaned[nk] = nv
            if cleaned:
                _ROLES_LEXICON.update(cleaned)
    except Exception:
        # optional file; ignore errors to preserve old behavior
        pass

_load_roles_lexicon()

def _lexicon_lookup(text: str) -> Optional[str]:
    """
    Try to find a canonical role by scanning text against lexicon variants.
    Returns the canonical string if a variant appears (longest match wins), else None.
    """
    if not _ROLES_LEXICON:
        return None
    t = f" {_norm_space(text)} "
    best_key = None
    for variant in _ROLES_LEXICON.keys():
        # word-ish boundary check by spaces (post-normalization)
        v = f" {variant} "
        if v in t:
            # pick the longest matching variant to avoid 'qa' beating 'qa engineer'
            if best_key is None or len(variant) > len(best_key):
                best_key = variant
    return _ROLES_LEXICON.get(best_key) if best_key else None

# ---------------------------
# New helpers (additive only)
# ---------------------------

# NOTE: keep a broad suffix list (singular) and allow plural via regex.
_TITLE_SUFFIXES = (
    "engineer|developer|scientist|analyst|manager|architect|specialist|consultant|"
    "researcher|designer|administrator|lead|head|officer|intern|lawyer|attorney|advocate|barrister"
)

# Canonical role families to help the backend do a strict role gate.
# Order matters (more specific before generic).
_ROLE_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(ui/?ux|ux/?ui)\s+designer(?:s|es)?\b",                "ui/ux designer"),
    (r"\bproduct\s+designer(?:s|es)?\b",                        "product designer"),
    (r"\b(web|visual|graphic)\s+designer(?:s|es)?\b",           "web designer"),
    (r"\bfront[\- ]?end\s+(developer|engineer)(?:s|es)?\b",     "frontend developer"),
    (r"\bback[\- ]?end\s+(developer|engineer)(?:s|es)?\b",      "backend developer"),
    (r"\bfull[\- ]?stack\s+(developer|engineer)(?:s|es)?\b",    "full stack developer"),
    (r"\bmobile\s+(developer|engineer)(?:s|es)?\b",             "mobile developer"),
    (r"\b(android|ios)\s+(developer|engineer)(?:s|es)?\b",      "mobile developer"),
    (r"\b(etl)\s+(developer|engineer)(?:s|es)?\b",              "etl developer"),
    (r"\bdevops\b|\bsre\b|\bsite reliability\b",                "devops/sre"),
    (r"\bqa\b|\bquality assurance\b|\btest(ing)?\s+engineer\b", "qa engineer"),
    (r"\bdata\s+scientist(?:s|es)?\b",                          "data scientist"),
    (r"\bdata\s+engineer(?:s|es)?\b",                           "data engineer"),
    (r"\b(machine learning|ml)\s+(engineer|scientist)(?:s|es)?\b","ml engineer"),
    (r"\bai\s+(engineer|scientist)(?:s|es)?\b",                 "ml engineer"),
    # ✅ Additive: Legal roles (for "advocate/lawyer" style prompts)
    (r"\b(advocate|lawyer|attorney|barrister)(?:s|es)?\b",      "advocate"),
]

def _normalize_role_family(title: Optional[str]) -> Optional[str]:
    """
    Map a free-text title to a canonical role family string (best-effort).
    Returns None if we cannot confidently classify.
    """
    if not title:
        return None

    # 0) Try lexicon direct first (exact/contained variant -> canonical)
    canon = _lexicon_lookup(title)
    if canon:
        return canon

    # 1) Pattern-based mapping
    t = title.lower().strip()
    for pattern, family in _ROLE_FAMILY_PATTERNS:
        if re.search(pattern, t):
            return family

    # 2) generic fallbacks if nothing matched but we have a known suffix
    if re.search(r"\bdesigner(?:s|es)?\b", t):
        return "designer"
    if re.search(r"\b(developer|engineer)(?:s|es)?\b", t):
        return "software developer"
    if re.search(r"\bscientist(?:s|es)?\b", t):
        return "scientist"
    if re.search(r"\banalyst(?:s|es)?\b", t):
        return "analyst"
    if re.search(r"\b(advocate|lawyer|attorney|barrister)(?:s|es)?\b", t):
        return "advocate"

    # 3) As a last attempt, scan the whole title via lexicon normalization again
    return _lexicon_lookup(t)

def _clean_list(text: str) -> List[str]:
    """
    Split comma/pipe/slash/plus/semicolon and also break on 'and' when used as list glue.
    Keep lowercased trimmed tokens, ignore empties.
    """
    parts = re.split(r"[,\|/;+]", text)
    items: List[str] = []
    for part in parts:
        subparts = re.split(r"\band\b", part, flags=re.IGNORECASE)
        for sp in subparts:
            t = re.sub(r"\s+", " ", sp).strip().lower()
            if t:
                items.append(t)
    return items

def _extract_title(prompt: str) -> Optional[str]:
    """
    Try to pick a professional title like 'web designer', 'ml engineer', etc.
    Strategy:
      1) Lexicon scan (best-effort, longest variant match).
      2) Longest phrase ending with a known title suffix (allow plural).
      3) Fallback: phrase after 'for' (short).
    """
    p = prompt.lower()

    # 1) Lexicon scan across the whole prompt
    lex = _lexicon_lookup(p)
    if lex:
        return lex

    # 2) Suffix-based longest match (allow plural 's' or 'es')
    matches = list(re.finditer(rf"\b([a-z][a-z ]*?)\s+((?:{_TITLE_SUFFIXES})(?:es|s)?)\b", p))
    if matches:
        best = max(matches, key=lambda m: len(m.group(0)))
        return re.sub(r"\s+", " ", best.group(0)).strip()

    # 3) Phrase after 'for'
    m = re.search(r"\bfor\s+([a-z][a-z ]{2,})\b", p)
    if m:
        chunk = m.group(1)
        chunk = re.split(r"\b(with|having|and|,|\.|;|in|where|which|that)\b", chunk)[0].strip()
        if 1 <= len(chunk.split()) <= 6:
            return chunk

    return None

def _extract_locations(prompt: str) -> List[str]:
    p = prompt.lower()
    locs: List[str] = []

    # patterns like: "in Lahore", "based in Karachi", etc.
    for pat in [
        r"\bbased in\s+([a-z ,/\-]+)",
        r"\blocated in\s+([a-z ,/\-]+)",
        r"\bfrom\s+([a-z ,/\-]+)",
        # guard: avoid 'experience in <tech>' by requiring delimiter or start
        r"(?:(?:^)|(?:\s))(?:in)\s+([a-z ,/\-]+)",
    ]:
        for m in re.finditer(pat, p):
            chunk = m.group(1)
            # stop at common non-location tokens
            chunk = re.split(r"\b(with|having|and|skills?|experience|projects?|for|using|on|of|,|\.|;)\b", chunk)[0]
            for item in _clean_list(chunk):
                if item and not re.fullmatch(r"(with|and|or|the|a|an|remote|onsite|on site|wfh)", item):
                    locs.append(item)

    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for x in locs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _extract_must_include_skills_clause(prompt: str) -> List[str]:
    """
    Detect explicit MUST include sections:
      - "must include Excel, PowerPoint and Flora"
      - "must have python, sql"
      - "required skills: a, b, c"
      - "should include ..."
    Returns a list of skills that must ALL be present.
    """
    p = prompt.lower()
    clauses = [
        r"\bmust include\s+([a-z0-9 .,+/|\-#]+)",
        r"\bmust have\s+([a-z0-9 .,+/|\-#]+)",
        r"\brequired(?: skills?)?\s*[:\-]?\s*([a-z0-9 .,+/|\-#]+)",
        r"\bshould include\s+([a-z0-9 .,+/|\-#]+)",
        r"\bneed to (?:have|include)\s+([a-z0-9 .,+/|\-#]+)",
    ]
    out: List[str] = []
    for pat in clauses:
        for m in re.finditer(pat, p):
            span = re.split(r"\b(and|but|however|except|,|\.|;)\b", m.group(1))[0]
            out += _clean_list(span)
    # de-dup
    seen = set()
    musts: List[str] = []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            musts.append(s)
    return musts

def _extract_skills(prompt: str) -> Tuple[List[str], List[str], List[str], bool]:
    """
    Returns (skills_all, skills_any, skills_exclude, is_strict_all)
    - skills_all: include-all list (AND)
    - skills_any: include-any list (OR)
    - skills_exclude: must NOT have
    - is_strict_all: True iff a 'must include' style clause was detected
    """
    p = prompt.lower()
    include_all: List[str] = []
    include_any: List[str] = []
    exclude: List[str] = []

    # strict "must include" takes priority (AND)
    must_include = _extract_must_include_skills_clause(p)
    is_strict_all = len(must_include) > 0

    # explicit skills sections (treated as AND if not explicitly 'or')
    for pat in [
        r"\bskills?\s*[:\-]\s*([a-z0-9 .,+/|\-#]+)",
        r"\btech(?:nologies| stack)?\s*[:\-]?\s*([a-z0-9 .,+/|\-#]+)",
    ]:
        for m in re.finditer(pat, p):
            include_all += _clean_list(m.group(1))

    # include lists after "with/ having / experience in"
    for pat in [
        r"\bwith\s+([a-z0-9 .,+/|\-#]+)",
        r"\bhaving\s+([a-z0-9 .,+/|\-#]+)",
        r"\bexperience in\s+([a-z0-9 .,+/|\-#]+)",
    ]:
        for m in re.finditer(pat, p):
            chunk = re.split(r"\b(and|but|however|where|which|that|,|\.|;)\b", m.group(1))[0]
            include_all += _clean_list(chunk)

    # exclude skills: "without X", "no X", "exclude X"
    for pat in [
        r"\bwithout\s+([a-z0-9 .,+/|\-#]+)",
        r"\bno\s+([a-z0-9 .,+/|\-#]+)",
        r"\bexclude\s+([a-z0-9 .,+/|\-#]+)",
    ]:
        for m in re.finditer(pat, p):
            chunk = re.split(r"\b(and|,|\.|;)\b", m.group(1))[0]
            exclude += _clean_list(chunk)

    # convert 'or' groups into skills_any (e.g., "python or r")
    for m in re.finditer(r"\b([a-z0-9\+\.# ]+?)\s+or\s+([a-z0-9\+\.# ]+)\b", p):
        include_any += _clean_list(m.group(1))
        include_any += _clean_list(m.group(2))

    # Merge strict 'must include' into all-AND bucket (but keep the strict flag)
    include_all = must_include + include_all

    # de-dup helpers
    def _dedup(lst: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in lst:
            x = x.strip().lower()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedup(include_all), _dedup(include_any), _dedup(exclude), is_strict_all

def _extract_projects_keywords(prompt: str) -> List[str]:
    """
    Extract project keywords if user names them, e.g.:
      - "projects: fraud detection, recommendation system"
      - "worked on credit scoring"
      - quoted phrases "time series forecasting"
    """
    p = prompt.lower()
    wants: List[str] = []
    # "projects: fraud detection, recommendation system"
    for pat in [r"\bprojects?\s*[:\-]\s*([a-z0-9 .,+/|\-#]+)",
                r"\bworked on\s+([a-z0-9 .,+/|\-#]+)"]:
        for m in re.finditer(pat, p):
            chunk = re.split(r"\b(and|,|\.|;)\b", m.group(1))[0]
            wants += _clean_list(chunk)

    # quoted phrases as project keywords
    for m in re.finditer(r"\"([a-z0-9 \-_/\.#\+]+)\"", p):
        wants.append(m.group(1).strip().lower())

    # de-dup
    seen = set()
    out: List[str] = []
    for x in wants:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _projects_required_flag(prompt: str) -> bool:
    """
    Detects if projects presence is required:
      - "should/must/need to have projects"
      - "projects required"
      - "should work on projects"
    """
    p = prompt.lower()
    return bool(
        re.search(r"\b(projects? (are )?required)\b", p)
        or re.search(r"\b(should|must|needs? to|need to)\s+(work on|have)\s+projects?\b", p)
        or re.search(r"\b(projects? section)\b", p)
    )

# ---------------------------
# Number words → numbers (for broader coverage)
# ---------------------------
_NUM_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

def _word_to_num(token: str) -> Optional[float]:
    """Convert 'four' / 'twenty-five' → numeric float."""
    t = token.strip().lower()
    if re.match(r"^\d+(?:\.\d+)?$", t):
        try:
            return float(t)
        except Exception:
            return None
    if t in _NUM_WORDS:
        return float(_NUM_WORDS[t])
    if "-" in t:
        parts = t.split("-")
        if all(p in _NUM_WORDS for p in parts):
            total = 0
            for p in parts:
                v = _NUM_WORDS[p]
                if v == 100 and total > 0:
                    total *= v
                else:
                    total += v
            return float(total)
    return None

def _tok_to_num(token: str) -> Optional[float]:
    """Helper that tries digits first, then number words."""
    token = token.strip().lower()
    if re.match(r"^\d+(?:\.\d+)?$", token):
        try:
            return float(token)
        except Exception:
            return None
    return _word_to_num(token)

def _find_num_after_phrase(p: str, phrase: str) -> Optional[float]:
    """
    Find a number or number-word after a phrase, like:
      phrase ... <num> (years|yrs)?
    """
    ph = re.escape(phrase)
    # allow a few words between phrase and number
    m = re.search(rf"{ph}\s+(?:[a-z ]+\s+)?([a-z0-9\-\.]+)\s*(?:years?|yrs?|yr|yoe)?\b", p)
    if not m:
        return None
    return _tok_to_num(m.group(1))

def _extract_experience(prompt: str) -> Dict[str, float]:
    """
    Supports:
      - "more than 3 years", "greater than 3 years", "3+ years", "at least 3 years" -> gte/gt
      - "less than 5 years", "under 5 years", "max 5 years", "at most 5" -> lte/lt
      - "between 3 and 5 years", "between 3 to 5 years", "3-5 years", "3–5 years" -> gte/lte
      - "= 4 years", "exactly 4 years" -> eq
      - "min 3", "max 7"
      - Symbols: <=, >=, <, >
      - ✅ Accepts number-words: "four years", "at least five", "between three and five"
      - ✅ Accepts phrase operators: "less than or equal to", "greater than or equal to"
      - ✅ Accepts 'yrs', 'yr', 'yoe'
    Returns:
      Either {"op":"<|>|=","years":X} or {"gte":A,"lte":B} or {"gte":X} / {"lte":Y}
    """
    p = prompt.lower()
    exp: Dict[str, float] = {}

    YR = r"(?:years?|yrs?|yr|yoe)"
    NUM = r"([a-z0-9\-\.]+)"  # allow words like "four" / "twenty-five"

    def num_of(group: str) -> Optional[float]:
        return _tok_to_num(group)

    # Symbols: <=, >=, <, >
    m = re.search(rf"\b>=\s*{NUM}\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            return {"g te".replace(" ", ""): float(n)}
    m = re.search(rf"\b<=\s*{NUM}\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            return {"lte": float(n)}
    m = re.search(rf"\b>\s*{NUM}\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            return {"op": ">", "years": float(n)}
    m = re.search(rf"\b<\s*{NUM}\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            return {"op": "<", "years": float(n)}

    # Phrase operators (<= and >=)
    for phrase in ["less than or equal to", "less than equal to", "at most", "no more than", "up to", "maximum", "max"]:
        n = _find_num_after_phrase(p, phrase)
        if n is not None:
            exp["lte"] = float(n)
            break
    for phrase in ["greater than or equal to", "at least", "minimum", "min", "not less than"]:
        n = _find_num_after_phrase(p, phrase)
        if n is not None:
            exp["gte"] = float(n)
            break

    # Strict < / > phrases
    for phrase in ["less than", "under", "below"]:
        n = _find_num_after_phrase(p, phrase)
        if n is not None:
            return {"op": "<", "years": float(n)}
    for phrase in ["greater than", "more than", "over", "above"]:
        n = _find_num_after_phrase(p, phrase)
        if n is not None:
            return {"op": ">", "years": float(n)}

    # between X and Y / between X to Y
    m = re.search(rf"\bbetween\s+{NUM}\s+(?:and|to)\s+{NUM}\s*{YR}?\b", p)
    if m:
        a = num_of(m.group(1))
        b = num_of(m.group(2))
        if a is not None and b is not None:
            lo, hi = (a, b) if a <= b else (b, a)
            return {"gte": float(lo), "lte": float(hi)}

    # range like "3-5 years" / "three-to five years" (hyphenated words supported)
    m = re.search(rf"\b{NUM}\s*(?:-|–|—|to)\s*{NUM}\s*{YR}?\b", p)
    if m:
        a = num_of(m.group(1))
        b = num_of(m.group(2))
        if a is not None and b is not None:
            lo, hi = (a, b) if a <= b else (b, a)
            return {"gte": float(lo), "lte": float(hi)}

    # exact = / equals / exactly
    m = re.search(rf"\b(exactly|equal(?:s)? to|=)\s*{NUM}\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(2)) if m.lastindex and m.lastindex >= 2 else num_of(m.group(1))
        if n is not None:
            return {"op": "=", "years": float(n)}

    # >= / 3+ years / 'at least' handled earlier, but keep a safety net
    m = re.search(rf"\b{NUM}\s*\+\s*{YR}?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            exp["gte"] = float(n)

    # explicit min/max X years (allow missing year token)
    m = re.search(rf"\bmin\s+{NUM}(?:\s*{YR})?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            exp["gte"] = float(n)
    m = re.search(rf"\bmax(?:imum)?\s+{NUM}(?:\s*{YR})?\b", p)
    if m:
        n = num_of(m.group(1))
        if n is not None:
            exp["lte"] = float(n)

    # simple "X yrs"/"X yoe"/"X years" with digits or words → treat as >= X
    m = re.search(rf"\b{NUM}\s*{YR}\b", p)
    if m and "gte" not in exp and "lte" not in exp and "op" not in exp:
        n = num_of(m.group(1))
        if n is not None:
            exp["gte"] = float(n)

    # "X+" (no unit) → treat as >= X
    m = re.search(rf"\b{NUM}\+\b", p)
    if m and "gte" not in exp:
        n = num_of(m.group(1))
        if n is not None:
            exp["gte"] = float(n)

    return exp

# ---------------------------
# Strong filtering (NEW additive fields)
# ---------------------------

def _extract_exact_phrases(prompt: str) -> List[str]:
    p = prompt
    phrases: List[str] = []

    # quoted "..." phrases (keep as-is, lowercased)
    for m in re.finditer(r"\"([^\"]{2,120})\"", p):
        ph = m.group(1).strip().lower()
        if ph:
            phrases.append(ph)

    # contains/containing/about/phrase
    for pat in [
        r"\bcontains?\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\bcontaining\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\babout\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\binclude(?:s|)\s+(?:phrase|text)\s+([a-z0-9 \-_/\.#\+]{2,120})",
    ]:
        for m in re.finditer(pat, p, flags=re.IGNORECASE):
            span = m.group(1)
            span = re.split(r"[.;,]| and | but | however ", span, flags=re.IGNORECASE)[0]
            span = re.sub(r"\s+", " ", span).strip().lower()
            if span:
                phrases.append(span)

    # normalize + de-dup
    seen = set()
    out: List[str] = []
    for ph in phrases:
        if ph and ph not in seen:
            seen.add(ph)
            out.append(ph)
    return out

def _extract_exclude_phrases(prompt: str) -> List[str]:
    p = prompt
    negs: List[str] = []
    for pat in [
        r"\bexclude(?:s|)\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\bno\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\bwithout\s+([a-z0-9 \-_/\.#\+]{2,120})",
        r"\bexcept\s+([a-z0-9 \-_/\.#\+]{2,120})",
    ]:
        for m in re.finditer(pat, p, flags=re.IGNORECASE):
            span = m.group(1)
            span = re.split(r"[.;,]| and | but | however ", span, flags=re.IGNORECASE)[0]
            span = re.sub(r"\s+", " ", span).strip().lower()
            if span:
                negs.append(span)

    seen = set()
    out: List[str] = []
    for ph in negs:
        if ph and ph not in seen:
            seen.add(ph)
            out.append(ph)
    return out

# — Education/university & degree constraints —
_DEGREE_TOKENS = [
    # legal & business
    "llb", "ll.m", "llm", "jd", "bar", "bar-at-law", "bar at law",
    "mba", "bba",
    # generic
    "bs", "b.s", "b.sc", "bsc", "ba", "b.a", "b.tech", "btech",
    "ms", "m.s", "m.sc", "msc", "ma", "m.a", "m.tech", "mtech",
    "phd", "dphil", "doctorate", "bachelors", "masters"
]

def _extract_education_requirements(prompt: str) -> Tuple[List[str], List[str]]:
    """
    Returns (schools_required, degrees_required).
    """
    p = prompt.lower()
    schools: List[str] = []
    degrees: List[str] = []

    # degree tokens presence anywhere
    for token in _DEGREE_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", p):
            degrees.append(token)

    # explicit “graduated from / from <school> / <Law School>”
    for pat in [
        r"\bgraduated from\s+([a-z][a-z0-9&\-\. ]{2,80})",
        r"\bgraduate of\s+([a-z][a-z0-9&\-\. ]{2,80})",
        r"\bfrom\s+([a-z][a-z0-9&\-\. ]{2,80})(?: university| college| law school)\b",
        r"\b([a-z][a-z0-9&\-\. ]{2,80})\s+(?:university|college|law school)\b",
    ]:
        for m in re.finditer(pat, p, flags=re.IGNORECASE):
            sch = m.group(1).strip().lower()
            sch = re.split(r"[.;,]| and | but | however ", sch, flags=re.IGNORECASE)[0]
            sch = re.sub(r"\s+", " ", sch).strip()
            if sch:
                schools.append(sch)

    # de-dup
    schools_dedup: List[str] = []
    seen = set()
    for s in schools:
        if s not in seen:
            seen.add(s)
            schools_dedup.append(s)

    degrees_dedup: List[str] = []
    seen.clear()
    for d in degrees:
        if d not in seen:
            seen.add(d)
            degrees_dedup.append(d)

    return schools_dedup, degrees_dedup

def _normalize_query_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize collected fields into keys expected by downstream.
    (Additive: we keep existing keys and supply a few helpful aliases.)
    """
    out: Dict[str, Any] = {
        "intent": "filter_cv",
        "query": parsed.get("query", ""),
    }
    if parsed.get("job_title"):
        out["job_title"] = parsed["job_title"]
        # helpful aliases for strict gating downstream
        out["job_title_normalized"] = parsed["job_title"]
        # ✅ aliases used by response builder's _collect_filter_intent
        out["title"] = parsed["job_title"]
        out["role"] = parsed["job_title"]
        out["position"] = parsed["job_title"]
    if parsed.get("role_family"):
        out["role_family"] = parsed["role_family"]
    if parsed.get("locations"):
        out["locations"] = parsed["locations"]
    if parsed.get("skills_all"):
        out["skills_all"] = parsed["skills_all"]
        # alias to emphasize MUST-have list (UI/ML may prefer this name)
        out["must_have_skills"] = parsed["skills_all"]
    if parsed.get("skills_any"):
        out["skills_any"] = parsed["skills_any"]
    if parsed.get("skills_exclude"):
        out["skills_exclude"] = parsed["skills_exclude"]
    if parsed.get("projects_keywords"):
        out["projects_keywords"] = parsed["projects_keywords"]
        # ✅ provide alternative keys some code paths may check
        out["projects"] = parsed["projects_keywords"]
        out["project_terms"] = parsed["projects_keywords"]
    if "projects_required" in parsed:
        out["projects_required"] = bool(parsed["projects_required"])
    if parsed.get("experience"):
        out["experience"] = parsed["experience"]
        # ✅ expose single-sided convenience aliases for downstream flexibility
        if isinstance(parsed["experience"], dict):
            op = parsed["experience"].get("op")
            yrs = parsed["experience"].get("years")
            if op == ">" and yrs is not None:
                out["experience_more_than"] = yrs
            if op == "<" and yrs is not None:
                out["experience_less_than"] = yrs
    if "skills_all_strict" in parsed:
        out["skills_all_strict"] = bool(parsed["skills_all_strict"])
    # convenience numeric bounds (additive; safely ignored if unused)
    if "min_years" in parsed and parsed["min_years"] is not None:
        out["min_years"] = parsed["min_years"]
        out["min_experience"] = parsed["min_years"]  # ✅ alias for response_builder
    if "max_years" in parsed and parsed["max_years"] is not None:
        out["max_years"] = parsed["max_years"]
        out["max_experience"] = parsed["max_years"]  # ✅ alias for response_builder

    # ✅ NEW additive normalizations for strong filtering
    if parsed.get("must_have_phrases"):
        out["must_have_phrases"] = parsed["must_have_phrases"]
    if parsed.get("exclude_phrases"):
        out["exclude_phrases"] = parsed["exclude_phrases"]
    if parsed.get("schools_required"):
        out["schools_required"] = parsed["schools_required"]
    if parsed.get("degrees_required"):
        out["degrees_required"] = parsed["degrees_required"]

    return out

# ---------------------------
# Main entry (expanded)
# ---------------------------

def parse_prompt(prompt: str) -> Dict[str, Any]:
    """
    Parses the user's prompt to detect intent + structured filters.
    """
    prompt = prompt.strip()

    # existing intents first
    if detect_usage_help(prompt):
        return {"intent": "usage_help", "query": prompt.lower()}

    if detect_show_all(prompt):
        return {"intent": "show_all", "query": prompt.lower()}

    # Structured extraction
    title = _extract_title(prompt)
    role_family = _normalize_role_family(title if title else prompt)  # pass prompt for lexicon fallback
    locations = _extract_locations(prompt)
    skills_all, skills_any, skills_exclude, skills_all_strict = _extract_skills(prompt)
    projects_keywords = _extract_projects_keywords(prompt)
    projects_required = _projects_required_flag(prompt)
    experience = _extract_experience(prompt)

    # ✅ NEW: Strong filters
    must_have_phrases = _extract_exact_phrases(prompt)
    exclude_phrases = _extract_exclude_phrases(prompt)
    schools_required, degrees_required = _extract_education_requirements(prompt)

    # derive min/max if present (additive convenience for scorers)
    min_years = None
    max_years = None
    if isinstance(experience, dict):
        if "gte" in experience:
            min_years = experience["gte"]
        if "lte" in experience:
            max_years = experience["lte"]
        if experience.get("op") == "=" and "years" in experience:
            min_years = max_years = experience["years"]

    parsed: Dict[str, Any] = {
        "query": prompt.lower(),
        "job_title": (title or role_family or None),
        "role_family": role_family or None,
        "locations": locations or None,
        "skills_all": skills_all or None,
        "skills_any": skills_any or None,
        "skills_exclude": skills_exclude or None,
        "skills_all_strict": skills_all_strict,  # boolean
        "projects_keywords": projects_keywords or None,
        "projects_required": projects_required,  # boolean
        "experience": experience or None,
        "min_years": min_years,
        "max_years": max_years,
        # NEW additive strong filters
        "must_have_phrases": must_have_phrases or None,
        "exclude_phrases": exclude_phrases or None,
        "schools_required": schools_required or None,
        "degrees_required": degrees_required or None,
    }

    # return normalized structure (intent = filter_cv by default)
    return _normalize_query_fields(parsed)
