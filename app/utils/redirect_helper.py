# app/utils/redirect_helper.py

from urllib.parse import urlencode

def build_redirect_url(filters: dict) -> str:
    """
    Builds the /filtered-results URL with query parameters from skills and experience.
    """
    base_url = "/filtered-results"
    params = {}

    if filters.get("skills"):
        params["skills"] = ",".join(filters["skills"])

    if filters.get("experience"):
        params["experience"] = filters["experience"]

    return f"{base_url}?{urlencode(params)}"
