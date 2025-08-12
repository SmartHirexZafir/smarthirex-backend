# app/utils/redirect_helper.py

from urllib.parse import urlencode, quote
import os
from typing import Optional, Iterable


def build_redirect_url(filters: dict) -> str:
    """
    Builds the /filtered-results URL with query parameters from skills and experience.

    - Accepts `skills` as list/tuple/set OR comma-separated string.
    - Avoids trailing '?' when there are no params.
    """
    base_url = "/filtered-results"
    params: dict[str, str] = {}

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
    if experience not in (None, ""):
        params["experience"] = str(experience)

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
