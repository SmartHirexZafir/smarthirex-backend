import json
import os
import difflib
from typing import Any, Dict, List, Optional, Set

from app.logic.ml_interface import get_semantic_matches  # ✅ semantic ranking source
from app.utils.redirect_helper import build_redirect_url


# ---------------------------
# Usage guide helpers (as-is)
# ---------------------------
async def load_usage_guide():
    path = os.path.join("app", "static", "usage_guide.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def fuzzy_match(prompt, guide):
    closest = difflib.get_close_matches(prompt.lower(), guide.keys(), n=1, cutoff=0.3)
    return guide[closest[0]] if closest else None


# ---------------------------
# Helpers: safe field getters
# ---------------------------
def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _as_set(items: Any) -> Set[str]:
    if items is None:
        return set()
    if isinstance(items, str):
        parts = [p.strip().lower() for p in items.split(",") if p.strip()]
        return set(parts)
    if isinstance(items, (list, tuple, set)):
        return set(_lower(x) for x in items if isinstance(x, str) and x.strip())
    return set()


def _get_job_title(resume: Dict[str, Any]) -> str:
    # include common aliases seen in parsed resumes
    return _lower(
        resume.get("title")
        or resume.get("job_title")
        or resume.get("current_title")
        or resume.get("headline")
        or resume.get("currentRole")          # 👈 added
        or resume.get("predicted_role")       # 👈 added
        or resume.get("category")             # 👈 added
        or ""
    )


def _get_location(resume: Dict[str, Any]) -> str:
    return _lower(
        resume.get("location")
        or resume.get("city")
        or resume.get("country")
        or resume.get("geo")
        or ""
    )


def _get_skills(resume: Dict[str, Any]) -> Set[str]:
    # Try multiple shapes
    skills = resume.get("skills") or resume.get("tech_skills") or resume.get("keywords")
    s = _as_set(skills)

    # also try free-text fields (light pickup; single-word tokens only)
    blob = " ".join(
        str(resume.get(k, "")) for k in ["summary", "about", "bio", "skills_text"]
    ).lower()
    for token in blob.replace("/", " ").replace("|", " ").split():
        if token.isalpha() and len(token) >= 2:
            s.add(token)
    return s


def _get_projects_text(resume: Dict[str, Any]) -> str:
    # Combine different project fields; handle dict/list shapes robustly
    parts: List[str] = []
    projects = resume.get("projects")

    if isinstance(projects, list):
        for p in projects:
            if isinstance(p, dict):
                parts.append(str(p.get("name", "")))
                parts.append(str(p.get("description", "")))
            else:
                parts.append(str(p))
    elif isinstance(projects, dict):
        parts.append(str(projects.get("name", "")))
        parts.append(str(projects.get("description", "")))
    elif isinstance(projects, str):
        parts.append(projects)

    for k in ["project_summary", "project_details", "portfolio", "experience"]:
        if resume.get(k):
            parts.append(str(resume[k]))
    return " ".join(parts).lower()


def _projects_present(projects_blob: str) -> bool:
    """
    Consider projects 'present' if we see the word or enough descriptive content.
    """
    if not projects_blob:
        return False
    if "project" in projects_blob:
        return True
    return len(projects_blob) > 60  # cheap heuristic used in ml_interface as well


def _get_total_years(resume: Dict[str, Any]) -> Optional[float]:
    """
    Try common numeric fields. Return None if not found.
    """
    for key in [
        "total_experience_years",
        "experience_years",
        "years_of_experience",
        "yoe",
        "overall_experience",
        "experience",
    ]:
        val = resume.get(key)
        try:
            if val is None:
                continue
            # handle strings like "3+", "4 years", "5.5"
            if isinstance(val, str):
                cleaned = "".join(ch for ch in val if (ch.isdigit() or ch == "." or ch == "-"))
                if cleaned in {"", ".", "-"}:
                    continue
                return float(cleaned)
            return float(val)
        except Exception:
            continue
    return None


# ---------------------------------
# Filters & flags
# ---------------------------------
def _match_title(resume_title: str, want_titles: Set[str]) -> bool:
    if not want_titles:
        return True
    # substring match any (semantic “related role” is handled earlier in ml_interface)
    return any(w in resume_title for w in want_titles)


def _match_location(resume_loc: str, want_locs: Set[str]) -> bool:
    if not want_locs:
        return True
    return any(w in resume_loc for w in want_locs)


def _match_skills(
    resume_skills: Set[str],
    include_all: Set[str],
    include_any: Set[str],
    exclude: Set[str],
    *,
    all_strict: bool
) -> bool:
    # must include ALL if 'must include' phrasing detected (all_strict=True),
    # otherwise still treat include_all as AND (safer for recruiter prompts).
    if include_all and not include_all.issubset(resume_skills):
        return False
    # must include at least one from include_any (if provided)
    if include_any and include_any.isdisjoint(resume_skills):
        return False
    # must NOT contain excluded
    if exclude and not exclude.isdisjoint(resume_skills):
        return False
    return True


def _match_projects(projects_blob: str, want_projects: Set[str], *, require_presence: bool) -> bool:
    # If presence is required (e.g., “should work on projects”), enforce existence first
    if require_presence and not _projects_present(projects_blob):
        return False
    # If specific project keywords are provided, require all of them
    if want_projects and not all(term in projects_blob for term in want_projects):
        return False
    return True


def _cmp_years(value: Optional[float], op: str, target: float) -> bool:
    if value is None:
        # Strict mode: unknown experience does NOT match any numeric filter
        return False
    if op in (">", "gt"):
        return value > target
    if op in (">=", "gte"):
        return value >= target
    if op in ("<", "lt"):
        return value < target
    if op in ("<=", "lte"):
        return value <= target
    if op in ("=", "==", "eq"):
        return value == target
    # unknown op -> pass
    return True


def _match_experience(years_value: Optional[float], parsed: Dict[str, Any], *, relax: bool = False) -> bool:
    """
    Supports shapes from intent_parser:
      - {"experience": {"op": ">", "years": 3}}
      - {"experience": {"gte": 3, "lte": 5}}
      - {"min_experience": 3, "max_experience": 5}
      - {"experience_more_than": 3}, {"experience_less_than": 5}
    If relax=True → always return True (experience ignored).
    If nothing found → True (no constraint).
    """
    if relax:
        return True

    exp = parsed.get("experience") or {}
    if isinstance(exp, dict):
        # Form 1: single op/years
        op = exp.get("op")
        yrs = exp.get("years")
        if op and yrs is not None:
            try:
                return _cmp_years(years_value, str(op).lower(), float(yrs))
            except Exception:
                pass
        # Form 2: range with gte/lte
        gte = exp.get("gte")
        lte = exp.get("lte")
        ok = True
        if gte is not None:
            ok = ok and _cmp_years(years_value, "gte", float(gte))
        if lte is not None:
            ok = ok and _cmp_years(years_value, "lte", float(lte))
        if (gte is not None) or (lte is not None):
            return ok

    # Form 3: min/max at root
    if parsed.get("min_experience") is not None or parsed.get("max_experience") is not None:
        ok = True
        if parsed.get("min_experience") is not None:
            ok = ok and _cmp_years(years_value, "gte", float(parsed["min_experience"]))
        if parsed.get("max_experience") is not None:
            ok = ok and _cmp_years(years_value, "lte", float(parsed["max_experience"]))
        return ok

    # Form 4: explicit more_than / less_than
    if parsed.get("experience_more_than") is not None:
        if not _cmp_years(years_value, "gt", float(parsed["experience_more_than"])):  # pyright: ignore
            return False
    if parsed.get("experience_less_than") is not None:
        if not _cmp_years(years_value, "lt", float(parsed["experience_less_than"])):  # pyright: ignore
            return False

    # No constraints found
    return True


def _collect_filter_intent(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize possibly-varied keys from intent_parser into a single filter spec.
    Does NOT error if keys are missing.
    """
    titles = _as_set(parsed.get("job_title") or parsed.get("title") or parsed.get("role") or parsed.get("position"))
    locations = _as_set(parsed.get("location") or parsed.get("locations"))
    # Skills buckets
    skills_all = _as_set(parsed.get("skills_all") or parsed.get("skills_required") or parsed.get("must_have_skills"))
    skills_any = _as_set(parsed.get("skills_any") or parsed.get("nice_to_have_skills") or parsed.get("optional_skills"))
    skills_exclude = _as_set(parsed.get("skills_exclude") or parsed.get("exclude_skills") or parsed.get("avoid_skills"))
    # Projects (as search terms)
    projects_terms = _as_set(parsed.get("projects") or parsed.get("project_terms") or parsed.get("projects_keywords"))

    return {
        "titles": titles,
        "locations": locations,
        "skills_all": skills_all,
        "skills_any": skills_any,
        "skills_exclude": skills_exclude,
        "projects_terms": projects_terms,
        "skills_all_strict": bool(parsed.get("skills_all_strict", False)),
        "projects_required": bool(parsed.get("projects_required", False)),
    }


def _has_any_filters(parsed: Dict[str, Any]) -> bool:
    spec = _collect_filter_intent(parsed)
    # presence of any non-empty set counts
    if spec["titles"] or spec["locations"] or spec["skills_all"] or spec["skills_any"] or spec["skills_exclude"] or spec["projects_terms"]:
        return True
    # experience counts as a filter as well (support all shapes)
    if parsed.get("experience"):
        return True
    if any(parsed.get(k) is not None for k in ("min_experience", "max_experience", "experience_more_than", "experience_less_than")):
        return True
    return False


def _apply_structured_filters(
    resumes: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    *,
    relax_experience: bool = False
) -> List[Dict[str, Any]]:
    """
    Apply AND-composed filters on top of semantic matches.
    Every provided constraint must pass.
    If relax_experience=True, experience condition is ignored (others still strict).
    """
    spec = _collect_filter_intent(parsed)

    out: List[Dict[str, Any]] = []
    for r in resumes:
        title_ok = _match_title(_get_job_title(r), spec["titles"])
        loc_ok = _match_location(_get_location(r), spec["locations"])
        skills_ok = _match_skills(
            _get_skills(r),
            spec["skills_all"],
            spec["skills_any"],
            spec["skills_exclude"],
            all_strict=spec["skills_all_strict"],
        )
        projects_ok = _match_projects(
            _get_projects_text(r),
            spec["projects_terms"],
            require_presence=spec["projects_required"],
        )
        exp_ok = _match_experience(_get_total_years(r), parsed, relax=relax_experience)

        if title_ok and loc_ok and skills_ok and projects_ok and exp_ok:
            out.append(r)
    return out


# ---------------------------
# Match summary for UI
# ---------------------------
def _summarize_matches(preview: List[Dict[str, Any]]) -> Dict[str, Any]:
    strict_count = 0
    close_count = 0
    for p in preview or []:
        if p.get("is_strict_match") is True or p.get("match_type") == "exact":
            strict_count += 1
        elif p.get("is_strict_match") is False or p.get("match_type") == "close":
            close_count += 1
        else:
            strict_count += 1  # default to strict if flags missing
    total = len(preview or [])
    return {
        "strictCount": strict_count,
        "closeCount": close_count,
        "total": total,
        "hasExact": strict_count > 0,
        "hasClose": close_count > 0 or (total > 0 and strict_count < total),
    }


# ---------------------------
# Main entry
# ---------------------------
async def build_response(parsed_data: dict) -> dict:
    intent = parsed_data.get("intent")
    # prefer original query if present else normalized_prompt (prevents blank UI text)
    prompt = (parsed_data.get("query") or parsed_data.get("normalized_prompt") or "").strip()

    if intent in ("filter_cv", "show_all"):
        # 1) Get semantic candidates (existing behavior preserved)
        resumes = await get_semantic_matches(
            prompt,
            owner_user_id=parsed_data.get("ownerUserId"),  # may be None; fine
            normalized_prompt=parsed_data.get("normalized_prompt"),
            keywords=parsed_data.get("keywords"),
        )

        redirect_url = build_redirect_url(parsed_data)

        # SHOW ALL → no structured filters (but keep original match flags)
        if intent == "show_all":
            count = len(resumes or [])
            reply_text = f"Showing {count} results for your query."
            return {
                "reply": reply_text,
                "redirect": redirect_url,
                "resumes_preview": resumes or [],
                "matchMeta": _summarize_matches(resumes or []),  # ✅ handy for UI header
                "ui": {"primaryMessage": reply_text, "query": prompt},
            }

        # 2) Apply strict structured filters if the parser provided any
        filters_present = _has_any_filters(parsed_data)
        filtered_resumes = _apply_structured_filters(resumes or [], parsed_data, relax_experience=False)

        if filtered_resumes:
            # Mark these as exact (if ml_interface didn't already)
            for it in filtered_resumes:
                if "is_strict_match" not in it:
                    it["is_strict_match"] = True
                if "match_type" not in it:
                    it["match_type"] = "exact"
            count = len(filtered_resumes)
            reply_text = f"Showing {count} results for your query."
            return {
                "reply": reply_text,
                "redirect": redirect_url,
                "resumes_preview": filtered_resumes,
                "matchMeta": _summarize_matches(filtered_resumes),
                "ui": {"primaryMessage": reply_text, "query": prompt},
            }

        # 3) If nothing found but filters were present, try a *soft* fallback:
        #    relax ONLY the experience filter (keep title/location/skills/projects strict)
        if filters_present:
            relaxed = _apply_structured_filters(resumes or [], parsed_data, relax_experience=True)
            if relaxed:
                for it in relaxed:
                    it["is_strict_match"] = False
                    it["match_type"] = "close"
                count = len(relaxed)
                reply_text = f"Showing {count} results for your query."
                return {
                    "reply": reply_text,
                    "redirect": redirect_url,
                    "resumes_preview": relaxed,
                    "matchMeta": _summarize_matches(relaxed),
                    "ui": {"primaryMessage": reply_text, "query": prompt},
                    "relaxedFlags": {"experience": True},  # ✅ optional hint for a subtle UI note (no “no exact” text)
                }

            # If still empty → return semantic list as last resort (still neutral wording)
            count_sem = len(resumes or [])
            for it in (resumes or []):
                it["is_strict_match"] = False
                it["match_type"] = "close"
            reply_text = f"Showing {count_sem} results for your query."
            return {
                "reply": reply_text,
                "redirect": redirect_url,
                "resumes_preview": resumes or [],
                "matchMeta": _summarize_matches(resumes or []),
                "ui": {"primaryMessage": reply_text, "query": prompt},
                "relaxedFlags": {"semanticFallback": True},
            }

        # No filters provided at all → just report semantic count
        count = len(resumes or [])
        reply_text = f"Showing {count} results for your query."
        return {
            "reply": reply_text,
            "redirect": redirect_url,
            "resumes_preview": resumes or [],
            "matchMeta": _summarize_matches(resumes or []),
            "ui": {"primaryMessage": reply_text, "query": prompt},
        }

    if intent == "usage_help":
        guide = await load_usage_guide()
        reply = await fuzzy_match(parsed_data.get("query", ""), guide)
        return {
            "reply": reply if reply else "Sorry, is feature ke bare me mujhe info nahi mili.",
            "redirect": None,
        }

    return {
        "reply": "Sorry, I couldn't understand your query.",
        "redirect": None,
    }
