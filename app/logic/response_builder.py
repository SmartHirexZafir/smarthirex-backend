# ✅ File: app/logic/response_builder.py

import json
import os
import difflib
from app.logic.ml_interface import get_ranked_resumes
from app.utils.redirect_helper import build_redirect_url

async def load_usage_guide():
    path = os.path.join("app", "static", "usage_guide.json")
    with open(path, "r") as f:
        return json.load(f)

async def fuzzy_match(prompt, guide):
    closest = difflib.get_close_matches(prompt.lower(), guide.keys(), n=1, cutoff=0.3)
    return guide[closest[0]] if closest else None

async def build_response(parsed_data: dict) -> dict:
    intent = parsed_data.get("intent")
    prompt = parsed_data.get("query", "")

    if intent == "filter_cv":
        # ✅ Extract expected ML filters safely
        filters = {
            "skills": parsed_data.get("skills", []),
            "experience": parsed_data.get("experience", 0),
            "job_role": parsed_data.get("job_role", ""),
            "project_keywords": parsed_data.get("project_keywords", [])
        }

        resumes = await get_ranked_resumes(filters)
        redirect_url = build_redirect_url(parsed_data)

        return {
            "reply": "Redirecting to filtered CV results page.",
            "redirect": redirect_url,
            "resumes_preview": resumes[:20]
        }

    if intent == "usage_help":
        guide = await load_usage_guide()
        reply = await fuzzy_match(prompt, guide)
        return {
            "reply": reply if reply else "Sorry, is feature ke bare me mujhe info nahi mili.",
            "redirect": None
        }

    return {
        "reply": "Sorry, I couldn't understand your query.",
        "redirect": None
    }
