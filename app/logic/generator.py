# app/logic/generator.py
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any

import openai
from app.models.auto_test_models import Candidate

openai.api_key = os.getenv("OPENAI_API_KEY")


def _extract_years(text: str) -> int:
    """
    Pulls the first integer from an experience string.
    Examples: "5 years", "3-4 yrs", "7+ years" -> 5, 3, 7.
    """
    if not text:
        return 0
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else 0


def _composition_for_experience(years: int) -> List[Dict[str, Any]]:
    """
    Returns a list describing the question composition by type.
    Each item: {"type": "mcq"|"code"|"scenario"}.
    """
    if years <= 2:
        # MCQs only
        return [{"type": "mcq"}] * 4
    if 3 <= years <= 6:
        # Mix: MCQs + Coding
        return [{"type": "mcq"}, {"type": "mcq"}, {"type": "mcq"}, {"type": "code"}]
    if years >= 10:
        # Scenario heavy
        return [{"type": "mcq"}, {"type": "scenario"}, {"type": "scenario"}]
    # Default balanced (previous behavior)
    return [{"type": "mcq"}, {"type": "mcq"}, {"type": "code"}, {"type": "scenario"}]


def _level_from_years(years: int) -> str:
    # Keep the original junior/senior notion for tone/difficulty
    return "junior" if years < 3 else "senior"


def _difficulty_sequence(n: int) -> List[str]:
    """
    Returns up to n difficulty labels in increasing order.
    (Used to tag MCQs Easy -> Hard even if model omits difficulty.)
    """
    base = ["easy", "medium", "hard", "expert"]
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(base[min(i, len(base) - 1)])
        i += 1
    return out[:n]


def _build_prompt(candidate: Candidate, comp: List[Dict[str, Any]]) -> str:
    """
    Builds a strict JSON generation prompt with explicit structure and counts.
    Also requests a 'difficulty' field for MCQs (easy/medium/hard/expert).
    """
    years = _extract_years(candidate.experience or "")
    level = _level_from_years(years)

    # Turn composition into human-readable spec for the model
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

    # Important: evaluator only auto-grades MCQs; for non-MCQ types we still require fields present.
    prompt = f"""
You are an HR test generator bot. Create a {level}-level assessment tailored to the candidate.

Candidate:
- Experience: {candidate.experience or "N/A"}
- Skills: {skills}
- Job Role: {role}

Composition:
{chr(10).join(spec_lines)}

Rules:
- Focus content on the candidate's role and skills.
- MCQs must include exactly 4 options and the correct option text in "correct_answer".
- MCQs should include a "difficulty" field among: "easy", "medium", "hard", "expert";
  order the MCQs from easy to hard.
- For "code" and "scenario" items, set "correct_answer" to null (still include the key).
- Do NOT include any explanations or commentary outside the JSON.
- Output must be a valid JSON array only (no code fences, no extra text).

JSON schema for each item:
{{
  "type": "mcq" | "code" | "scenario",
  "question": "string",
  "options": ["A","B","C","D"],     // only for mcq; omit for code/scenario or set to []
  "correct_answer": "B" | null,     // string for mcq, null for code/scenario
  "difficulty": "easy" | "medium" | "hard" | "expert"  // only for mcq (optional but preferred)
}}

Generate exactly {len(comp)} items with types in this order:
{[c["type"] for c in comp]}
"""
    return prompt.strip()


def _strip_code_fences(s: str) -> str:
    # If the model returns ```json ... ``` remove fences
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


def _normalize_items(items: List[Dict[str, Any]], comp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures each item has required keys and types, and matches the requested composition order/length.
    Adds/normalizes 'difficulty' for MCQs in increasing order if missing.
    """
    normalized: List[Dict[str, Any]] = []

    def _mk_mcq_stub(q: str = "Placeholder MCQ?") -> Dict[str, Any]:
        return {
            "type": "mcq",
            "question": q,
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
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
    mcq_difficulties = _difficulty_sequence(mcq_count)
    mcq_seen = 0

    # Take model items if present, but enforce length and order according to comp
    for i, spec in enumerate(comp):
        desired_type = spec.get("type", "mcq")
        candidate_item = items[i] if i < len(items) and isinstance(items[i], dict) else {}

        item_type = str(candidate_item.get("type", desired_type)).lower().strip()
        if item_type not in ("mcq", "code", "scenario"):
            item_type = desired_type

        qtext = candidate_item.get("question") or ""
        if not isinstance(qtext, str) or not qtext.strip():
            # build a stub if question missing
            normalized.append(creators[item_type]())
            # advance MCQ counter if needed
            if item_type == "mcq":
                mcq_seen += 1
            continue

        if item_type == "mcq":
            options = candidate_item.get("options") or []
            if not isinstance(options, list) or len(options) != 4:
                options = ["A", "B", "C", "D"]
            correct = candidate_item.get("correct_answer")
            if not isinstance(correct, str) or correct.strip() == "":
                correct = options[0]
            # difficulty (use model value if valid, else sequence)
            difficulty = str(candidate_item.get("difficulty") or "").lower().strip()
            if difficulty not in ("easy", "medium", "hard", "expert"):
                # fallback to our ordered sequence
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
            # code/scenario: ensure keys exist
            normalized.append({
                "type": item_type,
                "question": qtext.strip(),
                "options": [],
                "correct_answer": None,
            })

    return normalized


def _fallback_questions(candidate: Candidate, comp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic fallback set in case the API response isn't valid JSON.
    """
    role = candidate.job_role or "the role"
    skills = ", ".join(candidate.skills or []) or "core skills"

    out: List[Dict[str, Any]] = []
    mcq_seen = 0
    mcq_difficulties = _difficulty_sequence(sum(1 for c in comp if c.get("type") == "mcq"))

    for spec in comp:
        t = spec["type"]
        if t == "mcq":
            out.append({
                "type": "mcq",
                "question": f"In {role}, which of the following is MOST relevant to {skills}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "difficulty": mcq_difficulties[min(mcq_seen, len(mcq_difficulties) - 1)],
            })
            mcq_seen += 1
        elif t == "code":
            out.append({
                "type": "code",
                "question": f"Write a function related to {role} that demonstrates a key {skills} technique.",
                "options": [],
                "correct_answer": None,
            })
        else:
            out.append({
                "type": "scenario",
                "question": f"You are a {role}. Describe how you would handle a challenging situation involving {skills}.",
                "options": [],
                "correct_answer": None,
            })
    return out


def generate_test(candidate: Candidate) -> List[Dict[str, Any]]:
    """
    Generates a role/experience-tailored test.

    Preserves original behavior (OpenAI ChatCompletion with JSON response),
    but adjusts the number/types of questions based on experience:
      - <=2y: MCQs only (increasing difficulty)
      - 3–6y: MCQs + Coding
      - >=10y: Scenario heavy
      - default: balanced set (2 MCQ + 1 Code + 1 Scenario)
    """
    years = _extract_years((candidate.experience or "").strip())
    comp = _composition_for_experience(years)
    prompt = _build_prompt(candidate, comp)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response["choices"][0]["message"]["content"]
        content = _strip_code_fences(content)
        data = json.loads(content)

        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        normalized = _normalize_items(data, comp)
        return normalized

    except Exception:
        # Robust fallback that still respects the requested composition
        return _fallback_questions(candidate, comp)
