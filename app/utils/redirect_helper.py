# app/utils/redirect_helper.py

from urllib.parse import urlencode, quote
import os
from typing import Optional, Iterable, Any, List, Dict


def build_redirect_url(filters: dict) -> str:
    """
    Builds the /filtered-results URL with query parameters from skills and experience.

    - Accepts `skills` as list/tuple/set OR comma-separated string.
    - Avoids trailing '?' when there are no params.
    - ✅ Extended to include richer filter shapes used across the app:
        * skills_all / skills_any / skills_exclude / skills_all_strict
        * job_title / role_family / locations
        * projects_terms / projects_required
        * must_have_phrases / exclude_phrases
        * schools_required / degrees_required
        * keywords / query / normalized_prompt
        * experience as dict: {"op":..,"years":..} or {"gte":..,"lte":..}
          plus min_experience / max_experience / min_years / max_years
    """
    base_url = "/filtered-results"
    params: dict[str, str] = {}

    # ------------- helpers -------------
    def _listish_to_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            # split on commas for convenience
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return parts if parts else ([val.strip()] if val.strip() else [])
        return []

    def _put_csv(key: str, val: Any) -> None:
        items = _listish_to_list(val)
        if items:
            params[key] = ",".join(items)

    def _put_bool(key: str, val: Any) -> None:
        if isinstance(val, bool):
            params[key] = "1" if val else "0"

    def _put_str(key: str, val: Any) -> None:
        if val is None:
            return
        s = str(val).strip()
        if s != "" and s.lower() != "none":
            params[key] = s

    # ------------- legacy behavior (kept) -------------
    skills = filters.get("skills")
    if skills:
        if isinstance(skills, (list, tuple, set)):
            skills_str = ",".join(
                s.strip() for s in map(str, skills) if str(s).strip()
            )
        elif isinstance(skills, str):
            skills_str = ",".join(
                part.strip() for part in skills.split(",") if part.strip()
            )
        else:
            skills_str = ""
        if skills_str:
            params["skills"] = skills_str

    experience = filters.get("experience")
    if experience not in (None, "") and not isinstance(experience, dict):
        # keep legacy scalar/str experience param
        params["experience"] = str(experience)

    # ------------- new richer mappings (additive) -------------
    # Roles / title
    _put_str("title", filters.get("job_title") or filters.get("job_title_normalized") or filters.get("role_family"))

    # Locations
    _put_csv("locations", filters.get("locations") or filters.get("location"))

    # Skills buckets
    _put_csv("skills_all", filters.get("skills_all") or filters.get("must_have_skills"))
    _put_csv("skills_any", filters.get("skills_any"))
    _put_csv("skills_exclude", filters.get("skills_exclude"))
    _put_bool("skills_all_strict", filters.get("skills_all_strict"))

    # Projects
    _put_csv("projects", filters.get("projects_terms") or filters.get("projects_keywords"))
    _put_bool("projects_required", filters.get("projects_required"))

    # Strong phrase/education filters
    _put_csv("must_phrases", filters.get("must_have_phrases"))
    _put_csv("exclude_phrases", filters.get("exclude_phrases"))
    _put_csv("schools_required", filters.get("schools_required"))
    _put_csv("degrees_required", filters.get("degrees_required"))

    # Keywords + queries (for UI convenience)
    _put_csv("keywords", filters.get("keywords"))
    _put_str("query", filters.get("query"))
    _put_str("normalized_prompt", filters.get("normalized_prompt"))

    # Experience (structured shapes)
    exp_obj = experience if isinstance(experience, dict) else None
    if exp_obj:
        op = exp_obj.get("op")
        yrs = exp_obj.get("years")
        gte = exp_obj.get("gte")
        lte = exp_obj.get("lte")
        if op and yrs is not None:
            _put_str("exp_op", op)
            _put_str("exp_years", yrs)
        if gte is not None:
            _put_str("min_experience", gte)
        if lte is not None:
            _put_str("max_experience", lte)

    # Additional numeric bounds if present
    _put_str("min_experience", filters.get("min_experience") if filters.get("min_experience") is not None else filters.get("min_years"))
    _put_str("max_experience", filters.get("max_experience") if filters.get("max_experience") is not None else filters.get("max_years"))

    return f"{base_url}?{urlencode(params)}" if params else base_url


# ─────────────────────────────────────────────────────────────
# ✅ Added for Interview Scheduling (non-breaking additions)
# ─────────────────────────────────────────────────────────────

def _frontend_base_url(explicit: Optional[str] = None) -> str:
    """
    Resolve frontend base URL in this order:
    1) explicit argument,
    2) FRONTEND_BASE_URL env,
    3) WEB_BASE_URL env,
    4) http://localhost:3000 (default).
    """
    if explicit and explicit.strip():
        return explicit.rstrip("/")
    return (os.getenv("FRONTEND_BASE_URL")
            or os.getenv("WEB_BASE_URL")
            or "http://localhost:3000").rstrip("/")


def build_meeting_url(token: str, *, frontend_base: Optional[str] = None, path: str = "/meetings") -> str:
    """
    Build an absolute meeting URL for a given meeting token.

    Examples:
      build_meeting_url("abc123") ->
        http(s)://<frontend>/meetings/abc123

      build_meeting_url("abc123", frontend_base="https://app.example.com") ->
        https://app.example.com/meetings/abc123
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("token is required")

    # URL-encode token to be safe (handles any special chars)
    safe_token = quote(token.strip(), safe="")

    base = _frontend_base_url(frontend_base)
    # Ensure single slash between base and path, and between path and token
    normalized_path = "/" + path.strip("/")
    return f"{base}{normalized_path}/{safe_token}"
