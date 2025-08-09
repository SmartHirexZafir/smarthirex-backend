# app/logic/generator.py
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any, Optional

# --- OpenAI: support both old and new SDKs without breaking existing envs ---
NEW_SDK_AVAILABLE = False
try:
    # OpenAI Python v1.x
    from openai import OpenAI  # type: ignore
    NEW_SDK_AVAILABLE = True
except Exception:
    pass

import openai  # keep import for legacy paths

from app.models.auto_test_models import Candidate

# Optional: use centralized composition helpers if present
_COMPOSITION_HELPERS = False
try:
    # ✅ Use distinct aliases so we don't shadow local functions
    from app.logic.composition import (
        years_from_string as _years_from_string_helper,
        composition_for_experience as _composition_for_experience_helper,
    )
    _COMPOSITION_HELPERS = True
except Exception:
    _COMPOSITION_HELPERS = False

# API key for legacy SDK usage
openai.api_key = os.getenv("OPENAI_API_KEY")


# ------------------------
# Experience helpers
# ------------------------

def _extract_years(text: str) -> int:
    """
    Pulls the first integer from an experience string.
    Examples: "5 years", "3-4 yrs", "7+ years" -> 5, 3, 7.
    """
    if not text:
        return 0
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def _years_from_string(expr: Optional[str]) -> int:
    """
    Wrapper that uses composition.py if available, else a simple extractor.
    Critically, this does NOT recurse into itself.
    """
    if _COMPOSITION_HELPERS:
        try:
            return int(_years_from_string_helper(expr))  # type: ignore[arg-type]
        except Exception:
            return 0
    return _extract_years(expr or "")


def _composition_for_experience_internal(years: int, question_count: int) -> List[Dict[str, Any]]:
    """
    Internal, minimal composition builder (used only if composition.py isn't available).
    Mirrors the rules you requested:

      - <= 2 years  : MCQs only (easy→medium)
      - 3–6 years   : Mix of MCQs + Coding (medium→hard)
      - 7–9 years   : Balanced (MCQ + Code + Scenario)
      - >= 10 years : Scenario-heavy (advanced)

    MCQs listed first so the UI can ramp difficulty.
    """
    qn = max(1, int(question_count))

    if years <= 2:
        mcq = qn
        code = 0
        scenario = 0
    elif 3 <= years <= 6:
        # ~70% MCQ, ~30% code, at least 1 code if qn >= 3
        code = max(1 if qn >= 3 else 0, round(qn * 0.3))
        mcq = qn - code
        scenario = 0
    elif years >= 10:
        # scenario heavy; ensure at least 1 MCQ
        scenario = max(1, round(qn * 0.7))
        mcq = max(1, qn - scenario)
        code = 0
    else:
        # 7–9 yrs balanced
        mcq = max(1, round(qn * 0.5))
        code = 1 if qn >= 3 else 0
        scenario = max(1 if qn >= 4 else 0, qn - (mcq + code))

    comp: List[Dict[str, Any]] = []
    comp += [{"type": "mcq"}] * mcq
    comp += [{"type": "code"}] * code
    comp += [{"type": "scenario"}] * scenario
    # Trim/pad (safety)
    return comp[:qn] if len(comp) >= qn else (comp + [{"type": "mcq"}] * (qn - len(comp)))


def _composition_for_experience(years: int, question_count: int) -> List[Dict[str, Any]]:
    if _COMPOSITION_HELPERS:
        try:
            return _composition_for_experience_helper(years=years, question_count=question_count)  # type: ignore[misc]
        except Exception:
            pass
    return _composition_for_experience_internal(years, question_count)


def _level_from_years(years: int) -> str:
    # tone hint for prompt
    return "junior" if years < 3 else "senior"


def _difficulty_sequence(n: int, years: int) -> List[str]:
    """
    Returns n difficulty labels. We bias by experience:
      - <=2y: mostly easy→medium
      - 3–6y: medium→hard
      - >=10y: hard→expert
    """
    if years <= 2:
        seq = ["easy", "easy", "medium", "medium", "hard", "hard", "expert"]
    elif 3 <= years <= 6:
        seq = ["medium", "medium", "hard", "hard", "expert"]
    elif years >= 10:
        seq = ["hard", "hard", "expert", "expert"]
    else:
        seq = ["easy", "medium", "hard", "expert"]
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(seq[min(i, len(seq) - 1)])
        i += 1
    return out[:n]


# ------------------------
# Prompt & normalization
# ------------------------

def _build_prompt(candidate: Candidate, comp: List[Dict[str, Any]]) -> str:
    """
    Builds a strict JSON generation prompt with explicit structure and counts.
    """
    years = _years_from_string(candidate.experience)
    level = _level_from_years(years)

    counts = {"mcq": 0, "code": 0, "scenario": 0}
    for c in comp:
        t = c.get("type", "mcq")
        counts[t] = counts.get(t, 0) + 1

    spec_lines = []
    if counts["mcq"]:
        spec_lines.append(f"- {counts['mcq']} MCQ(s) (in increasing difficulty)")
    if counts["code"]:
        spec_lines.append(f"- {counts['code']} Coding question(s)")
    if counts["scenario"]:
        spec_lines.append(f"- {counts['scenario']} Scenario-based question(s)")

    role = candidate.job_role or "General"
    skills = ", ".join(candidate.skills or []) or "general skills"

    prompt = f"""
You are an assessment generator. Create a {level}-level test tailored to the candidate.

Candidate:
- Experience: {candidate.experience or "N/A"}
- Skills: {skills}
- Job Role: {role}

Composition:
{chr(10).join(spec_lines)}

Rules:
- Focus content on the candidate's role and skills.
- MCQs must include exactly 4 realistic option texts (not "A/B/C/D", not placeholders).
- The correct option text must be in "correct_answer".
- MCQs should include a "difficulty" field among: "easy", "medium", "hard", "expert";
  order MCQs from easier to harder based on the candidate's experience.
- For "code" and "scenario" items, include "correct_answer": null (still include the key) and omit "options".
- Do NOT include any commentary outside the JSON.
- Output must be a valid pure JSON array only.

JSON schema for each item:
{{
  "type": "mcq" | "code" | "scenario",
  "question": "string",
  "options": ["text1","text2","text3","text4"],   // only for mcq
  "correct_answer": "one of the options" | null,  // string for mcq, null for code/scenario
  "difficulty": "easy" | "medium" | "hard" | "expert"  // for mcq
}}

Generate exactly {len(comp)} items with types in this order:
{[c["type"] for c in comp]}
"""
    return prompt.strip()


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


# Simple role-aware option synthesizer to replace "A/B/C/D" if needed
def _synthesize_options(role: str, skills: List[str], question: str) -> List[str]:
    role_l = (role or "").lower()
    skill_blob = " ".join((skills or [])).lower()

    bank: List[List[str]] = []

    if "data" in role_l or "ml" in role_l or "ai" in role_l or "analyst" in role_l:
        bank.append([
            "Building and validating machine learning models",
            "Cleaning and preprocessing datasets",
            "Creating dashboards with BI tools",
            "Setting up REST API gateways",
        ])
        bank.append([
            "Using version control with Git",
            "Writing SQL queries for analysis",
            "Hyperparameter tuning in TensorFlow/PyTorch",
            "Designing responsive UI components",
        ])
    if "web" in role_l or "frontend" in role_l or "react" in role_l:
        bank.append([
            "Managing state with React hooks",
            "Optimizing bundle size and code-splitting",
            "Creating accessible UI components",
            "Training convolutional neural networks",
        ])
    if "backend" in role_l or "server" in role_l or "api" in role_l or "python" in skill_blob:
        bank.append([
            "Designing RESTful endpoints",
            "Writing unit tests and integration tests",
            "Configuring database schemas and indexes",
            "Adding CSS animations for hover states",
        ])

    if not bank:
        bank.append([
            "Planning solution architecture",
            "Collaborating with cross-functional teams",
            "Writing clean, maintainable code",
            "Conducting user interviews and surveys",
        ])

    # pick a group deterministically by question hash to avoid random flicker
    seed = abs(hash(question)) % len(bank)
    opts = bank[seed].copy()

    # Ensure four unique strings
    opts = [str(x) for x in opts][:4]
    while len(opts) < 4:
        opts.append(f"Relevant option {len(opts)+1}")

    return opts[:4]


def _normalize_items(
    items: List[Dict[str, Any]],
    comp: List[Dict[str, Any]],
    candidate: Candidate,
    years: int,
) -> List[Dict[str, Any]]:
    """
    Ensures each item has required keys/types and matches the requested composition order/length.
    Fills in real MCQ options and difficulty if missing.
    """
    normalized: List[Dict[str, Any]] = []

    def _mk_mcq_stub(q: str = "Which option is most appropriate?") -> Dict[str, Any]:
        opts = _synthesize_options(candidate.job_role or "Role", candidate.skills or [], q)
        return {
            "type": "mcq",
            "question": q,
            "options": opts,
            "correct_answer": opts[0],
            "difficulty": "easy",
        }

    def _mk_code_stub(q: str = "Write a function that ...") -> Dict[str, Any]:
        return {
            "type": "code",
            "question": q,
            "options": [],
            "correct_answer": None,
        }

    def _mk_scenario_stub(q: str = "Describe how you would handle ...") -> Dict[str, Any]:
        return {
            "type": "scenario",
            "question": q,
            "options": [],
            "correct_answer": None,
        }

    creators = {
        "mcq": _mk_mcq_stub,
        "code": _mk_code_stub,
        "scenario": _mk_scenario_stub,
    }

    # Prepare difficulty defaults for the number of MCQs requested
    mcq_count = sum(1 for c in comp if c.get("type") == "mcq")
    mcq_difficulties = _difficulty_sequence(mcq_count, years)
    mcq_seen = 0

    role = candidate.job_role or "Role"
    skills = candidate.skills or []

    for i, spec in enumerate(comp):
        desired_type = spec.get("type", "mcq")
        candidate_item = items[i] if i < len(items) and isinstance(items[i], dict) else {}

        item_type = str(candidate_item.get("type", desired_type)).lower().strip()
        if item_type not in ("mcq", "code", "scenario"):
            item_type = desired_type

        qtext = candidate_item.get("question") or ""
        if not isinstance(qtext, str) or not qtext.strip():
            # missing question → build stub
            normalized.append(creators[item_type]())
            if item_type == "mcq":
                mcq_seen += 1
            continue

        if item_type == "mcq":
            options = candidate_item.get("options") or []
            # If options are missing, not 4, or look like A/B/C/D → synthesize
            def looks_like_letters(arr: List[Any]) -> bool:
                try:
                    joined = " ".join([str(x).strip().upper() for x in arr])
                    return joined in ("A B C D", "A,B,C,D")
                except Exception:
                    return False

            if not isinstance(options, list) or len(options) != 4 or looks_like_letters(options):
                options = _synthesize_options(role, skills, qtext)

            # correct answer must be a string and belong to options
            correct = candidate_item.get("correct_answer")
            if not isinstance(correct, str) or correct.strip() == "" or correct.strip() not in options:
                correct = options[0]

            # difficulty (use model value if valid, else sequence)
            difficulty = str(candidate_item.get("difficulty") or "").lower().strip()
            if difficulty not in ("easy", "medium", "hard", "expert"):
                difficulty = mcq_difficulties[min(mcq_seen, len(mcq_difficulties) - 1)]

            normalized.append({
                "type": "mcq",
                "question": qtext.strip(),
                "options": options,
                "correct_answer": correct.strip(),
                "difficulty": difficulty,
            })
            mcq_seen += 1
        else:
            normalized.append({
                "type": item_type,
                "question": qtext.strip(),
                "options": [],
                "correct_answer": None,
            })

    return normalized


def _fallback_questions(candidate: Candidate, comp: List[Dict[str, Any]], years: int) -> List[Dict[str, Any]]:
    """
    Deterministic fallback set in case the API response isn't valid JSON.
    Creates realistic option texts for MCQs (not A/B/C/D).
    """
    role = candidate.job_role or "the role"
    skills = candidate.skills or []

    out: List[Dict[str, Any]] = []
    mcq_seen = 0
    mcq_difficulties = _difficulty_sequence(sum(1 for c in comp if c.get("type") == "mcq"), years)

    for spec in comp:
        t = spec["type"]
        if t == "mcq":
            qtext = f"In {role}, which of the following is MOST relevant to {', '.join(skills) or 'core skills'}?"
            opts = _synthesize_options(role, skills, qtext)
            out.append({
                "type": "mcq",
                "question": qtext,
                "options": opts,
                "correct_answer": opts[0],
                "difficulty": mcq_difficulties[min(mcq_seen, len(mcq_difficulties) - 1)],
            })
            mcq_seen += 1
        elif t == "code":
            out.append({
                "type": "code",
                "question": f"Write a function related to {role} that demonstrates a key {', '.join(skills) or 'technique'}.",
                "options": [],
                "correct_answer": None,
            })
        else:
            out.append({
                "type": "scenario",
                "question": f"You are a {role}. Describe how you would handle a complex situation involving {', '.join(skills) or 'your core responsibilities'}.",
                "options": [],
                "correct_answer": None,
            })
    return out


# ------------------------
# OpenAI call (new SDK first, fallback to legacy)
# ------------------------

def _call_openai(prompt: str) -> str:
    """
    Returns raw string content from OpenAI using whichever SDK is available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Prefer new SDK if present
    if NEW_SDK_AVAILABLE:
        client = OpenAI(api_key=api_key)  # type: ignore
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""

    # Legacy path
    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp["choices"][0]["message"]["content"]


# ------------------------
# Public entry
# ------------------------

def generate_test(candidate: Candidate, question_count: int = 4) -> List[Dict[str, Any]]:
    """
    Generates a role/experience-tailored test.

    Enhancements:
      - Experience-aware composition for ANY question_count.
      - Ensures real MCQ options (no bare A/B/C/D).
      - Compatible with both OpenAI SDKs; robust fallback.

    Args:
        candidate: Candidate model
        question_count: desired number of questions (default 4 for backward compatibility)

    Returns:
        List of normalized question dicts compatible with your evaluator/UI.
    """
    years = _years_from_string((candidate.experience or "").strip())
    comp = _composition_for_experience(years, question_count)
    prompt = _build_prompt(candidate, comp)

    try:
        content = _call_openai(prompt)
        content = _strip_code_fences(content)
        data = json.loads(content)

        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        normalized = _normalize_items(data, comp, candidate, years)
        return normalized

    except Exception:
        # Robust fallback that still respects the requested composition
        return _fallback_questions(candidate, comp, years)
