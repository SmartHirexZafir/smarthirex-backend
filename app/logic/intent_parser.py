# ✅ File: app/logic/intent_parser.py

import re
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------
# Existing logic (unchanged)
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
        "saare candidate"
    ]

# ---------------------------
# New helpers (additive only)
# ---------------------------

_TITLE_SUFFIXES = (
    "engineer|developer|scientist|analyst|manager|architect|specialist|consultant|"
    "researcher|designer|administrator|lead|head|officer|intern"
)

def _clean_list(text: str) -> List[str]:
    """
    Split comma/pipe/slash/plus/semicolon and also break on 'and' when used as list glue.
    Keep lowercased trimmed tokens, ignore empties.
    """
    # First split by common delimiters
    parts = re.split(r"[,\|/;+]", text)
    items: List[str] = []
    for part in parts:
        # Then split by ' and ' if it looks like a list
        subparts = re.split(r"\band\b", part, flags=re.IGNORECASE)
        for sp in subparts:
            t = re.sub(r"\s+", " ", sp).strip().lower()
            if t:
                items.append(t)
    return items

def _extract_title(prompt: str) -> Optional[str]:
    """
    Try to pick a professional title like 'data scientist', 'ml engineer', etc.
    Strategy:
      1) Look for longest phrase ending with a known title suffix.
      2) Fallback: phrase after 'for' (short).
    """
    p = prompt.lower()

    # 1) Suffix-based longest match
    matches = list(re.finditer(rf"\b([a-z][a-z ]*?)\s+({_TITLE_SUFFIXES})\b", p))
    if matches:
        best = max(matches, key=lambda m: len(m.group(0)))
        return re.sub(r"\s+", " ", best.group(0)).strip()

    # 2) Phrase after 'for'
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

    # patterns like: "in Lahore", "based in Karachi", "from Islamabad", "located in Dubai, UAE"
    for pat in [
        r"\bbased in\s+([a-z ,/\-]+)",
        r"\blocated in\s+([a-z ,/\-]+)",
        r"\bfrom\s+([a-z ,/\-]+)",
        r"\bin\s+([a-z ,/\-]+)",
    ]:
        for m in re.finditer(pat, p):
            chunk = m.group(1)
            chunk = re.split(r"\b(with|having|and|skills?|experience|projects?|for|,|\.|;)\b", chunk)[0]
            for item in _clean_list(chunk):
                # keep city/country tokens; ignore generic words
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

def _extract_experience(prompt: str) -> Dict[str, float]:
    """
    Supports:
      - "more than 3 years", "greater than 3 years", "3+ years", "at least 3 years" -> gte/gt
      - "less than 5 years", "under 5 years", "max 5 years", "at most 5" -> lte/lt
      - "between 3 and 5 years", "between 3 to 5 years", "3-5 years", "3–5 years" -> gte/lte
      - "= 4 years", "exactly 4 years" -> eq
      - "min 3", "max 7"
      - Symbols: <=, >=, <, >
    Returns:
      Either {"op":"<|>|=","years":X} or {"gte":A,"lte":B} or {"gte":X} / {"lte":Y}
    """
    p = prompt.lower()
    exp: Dict[str, float] = {}

    # Symbols: <=, >=, <, >
    m = re.search(r"\b>=\s*(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"gte": float(m.group(1))}
    m = re.search(r"\b<=\s*(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"lte": float(m.group(1))}
    m = re.search(r"\b>\s*(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"op": ">", "years": float(m.group(1))}
    m = re.search(r"\b<\s*(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"op": "<", "years": float(m.group(1))}

    # between X and Y / between X to Y
    m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return {"gte": lo, "lte": hi}

    # range like "3-5 years" / "3–5 years" / "3 — 5 years"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return {"gte": lo, "lte": hi}

    # >= / at least / minimum / 3+ years
    if re.search(r"\b(at least|min(?:imum)?)\b", p) or re.search(r"\b\d+\s*\+\s*years?\b", p):
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\b", p)
        if m:
            exp["gte"] = float(m.group(1))
            # keep parsing in case an upper bound is present

    # > / greater than / more than / over
    m = re.search(r"\b(greater than|more than|over)\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"op": ">", "years": float(m.group(2))}

    # <= / at most / maximum / up to
    m = re.search(r"\b(at most|max(?:imum)?|up to)\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        exp["lte"] = float(m.group(2))

    # < / less than / under / below
    m = re.search(r"\b(less than|under|below)\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"op": "<", "years": float(m.group(2))}

    # exact = / equals / exactly
    m = re.search(r"\b(exactly|equal(?:s)? to|=)\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        return {"op": "=", "years": float(m.group(2))}

    # explicit min/max X years
    m = re.search(r"\bmin\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        exp["gte"] = float(m.group(1))
    m = re.search(r"\bmax\s+(\d+(?:\.\d+)?)\s*years?\b", p)
    if m:
        exp["lte"] = float(m.group(1))

    return exp

def _normalize_query_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize collected fields into keys expected by downstream.
    """
    out: Dict[str, Any] = {
        "intent": "filter_cv",
        "query": parsed.get("query", ""),
    }
    if parsed.get("job_title"):
        out["job_title"] = parsed["job_title"]
    if parsed.get("locations"):
        out["locations"] = parsed["locations"]
    if parsed.get("skills_all"):
        out["skills_all"] = parsed["skills_all"]
    if parsed.get("skills_any"):
        out["skills_any"] = parsed["skills_any"]
    if parsed.get("skills_exclude"):
        out["skills_exclude"] = parsed["skills_exclude"]
    if parsed.get("projects_keywords"):
        out["projects_keywords"] = parsed["projects_keywords"]
    if "projects_required" in parsed:
        out["projects_required"] = bool(parsed["projects_required"])
    if parsed.get("experience"):
        out["experience"] = parsed["experience"]
    if "skills_all_strict" in parsed:
        out["skills_all_strict"] = bool(parsed["skills_all_strict"])
    return out

# ---------------------------
# Main entry (expanded)
# ---------------------------

def parse_prompt(prompt: str) -> Dict[str, Any]:
    """
    Parses the user's prompt to detect intent + structured filters.

    Returns (example):
    {
      "intent": "filter_cv",
      "query": "...",
      "job_title": "data scientist",
      "locations": ["karachi","pakistan"],
      "skills_all": ["excel","powerpoint","flora"],
      "skills_all_strict": true,          # <- detected from “must include”
      "skills_any": ["python","r"],
      "skills_exclude": ["php"],
      "projects_required": true,          # <- detected from wording
      "projects_keywords": ["fraud detection"],
      "experience": {"gte": 7, "lte": 9}  # or {"op": "<=", "years": 5}, etc.
    }
    """
    prompt = prompt.strip()

    # existing intents first
    if detect_usage_help(prompt):
        return {"intent": "usage_help", "query": prompt.lower()}

    if detect_show_all(prompt):
        return {"intent": "show_all", "query": prompt.lower()}

    # Structured extraction
    title = _extract_title(prompt)
    locations = _extract_locations(prompt)
    skills_all, skills_any, skills_exclude, skills_all_strict = _extract_skills(prompt)
    projects_keywords = _extract_projects_keywords(prompt)
    projects_required = _projects_required_flag(prompt)
    experience = _extract_experience(prompt)

    parsed: Dict[str, Any] = {
        "query": prompt.lower(),
        "job_title": title or None,
        "locations": locations or None,
        "skills_all": skills_all or None,
        "skills_any": skills_any or None,
        "skills_exclude": skills_exclude or None,
        "skills_all_strict": skills_all_strict,  # boolean
        "projects_keywords": projects_keywords or None,
        "projects_required": projects_required,  # boolean
        "experience": experience or None,
    }

    # return normalized structure (intent = filter_cv by default)
    return _normalize_query_fields(parsed)
