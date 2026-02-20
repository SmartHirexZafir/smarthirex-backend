# app/logic/generator.py
from __future__ import annotations

import json
import os
import re
import random
import hashlib
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
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return 0
    try:
        return max(0, int(float(m.group(1))))
    except Exception:
        return 0


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


def _build_composition_from_counts(
    mcq_count: int = 0,
    scenario_count: int = 0,
    code_count: int = 0,
) -> List[Dict[str, Any]]:
    """
    Build composition from explicit counts provided by the sender.
    Difficulty is handled separately from candidate experience.
    """
    comp: List[Dict[str, Any]] = []
    comp += [{"type": "mcq"}] * max(0, int(mcq_count))
    comp += [{"type": "code"}] * max(0, int(code_count))
    comp += [{"type": "scenario"}] * max(0, int(scenario_count))
    return comp


def _build_composition(
    mcq_count: int = 0,
    scenario_count: int = 0,
    code_count: int = 0,
    question_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build composition from explicit counts (preferred) or fallback to question_count.
    If composition counts are provided, use them. Otherwise, use question_count for backward compatibility.
    """
    if mcq_count > 0 or scenario_count > 0 or code_count > 0:
        # Use explicit composition
        return _build_composition_from_counts(mcq_count, scenario_count, code_count)
    
    # Fallback: backward compatibility with question_count
    if question_count and question_count > 0:
        # Default: all MCQs for backward compatibility
        return [{"type": "mcq"}] * question_count
    
    # Default fallback
    return [{"type": "mcq"}] * 4


def _level_from_years(years: int) -> str:
    """
    Difficulty bands required by product policy.
    0-2 years -> Beginner, 2-5 -> Intermediate, 5+ -> Advanced.
    """
    if years < 2:
        return "beginner"
    if years < 5:
        return "intermediate"
    return "advanced"


def _difficulty_sequence(n: int, level: str = "intermediate") -> List[str]:
    """
    Returns n difficulty labels in a progressive sequence,
    adapted to the candidate experience band.
    """
    level = (level or "intermediate").strip().lower()
    if level == "beginner":
        seq = ["easy", "easy", "medium", "medium"]
    elif level == "advanced":
        seq = ["hard", "hard", "expert", "expert"]
    else:
        seq = ["medium", "medium", "hard", "expert"]
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(seq[min(i % len(seq), len(seq) - 1)])
        i += 1
    return out[:n]


def _seed_to_int(seed: Optional[Any]) -> int:
    """
    Convert any seed-like value into a stable integer.
    """
    if seed is None:
        return int.from_bytes(os.urandom(8), "big", signed=False)
    if isinstance(seed, int):
        return abs(seed)
    raw = str(seed).strip()
    if not raw:
        return int.from_bytes(os.urandom(8), "big", signed=False)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _prompt_variation(seed_int: int) -> str:
    """
    Deterministic prompt variation so repeated attempts get different structures.
    """
    variants = [
        "Prioritize real-world troubleshooting and applied reasoning.",
        "Prioritize architecture trade-offs and decision-making clarity.",
        "Prioritize debugging, failure analysis, and recovery strategies.",
        "Prioritize optimization, performance, and scalability thinking.",
        "Prioritize practical implementation details and production constraints.",
        "Prioritize security, reliability, and edge-case handling.",
    ]
    return variants[seed_int % len(variants)]


# ------------------------
# Prompt & normalization
# ------------------------

def _build_prompt(
    candidate: Candidate,
    comp: List[Dict[str, Any]],
    previous_questions: Optional[List[str]] = None,
    variation_hint: Optional[str] = None,
) -> str:
    """
    Builds a strict JSON generation prompt with explicit structure and counts.
    Includes uniqueness enforcement to avoid duplicate questions.
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

    uniqueness_note = ""
    if previous_questions:
        uniqueness_note = f"""
CRITICAL: The following questions have already been used for this candidate. DO NOT repeat or recycle them:
{chr(10).join(f"- {q}" for q in previous_questions[:10])}  # Show max 10 to avoid prompt bloat

You MUST generate completely NEW, UNIQUE questions that are different in content, structure, and focus.
"""

    prompt = f"""
You are an assessment generator. Create a {level}-level test STRICTLY tailored to the candidate's job role.

Candidate:
- Job Role: {role}  ⚠️ THIS IS THE PRIMARY FOCUS - ALL QUESTIONS MUST BE ROLE-SPECIFIC
- Experience: {candidate.experience or "N/A"}
- Skills: {skills}
- Target Difficulty Band: {level}

CRITICAL ROLE ENFORCEMENT:
- The candidate's job role is: "{role}"
- EVERY SINGLE QUESTION must be DIRECTLY related to this specific job role
- If the role is "Doctor", generate ONLY medical, healthcare, clinical, diagnostic questions
- If the role is "Software Engineer", generate ONLY programming, software development, technical questions
- If the role is "Data Scientist", generate ONLY data analysis, machine learning, statistics questions
- NO generic questions, NO irrelevant topics, NO mismatched content
- Each question must test knowledge, skills, or scenarios specific to "{role}"
- If you cannot create role-specific questions, DO NOT generate generic questions - focus on the role

Composition:
{chr(10).join(spec_lines)}
{uniqueness_note}
Rules:
- ⚠️ MANDATORY: ALL questions must be SPECIFICALLY about "{role}" - no exceptions
- Focus content EXCLUSIVELY on the candidate's job role: {role}
- Variation directive: {variation_hint or "Use diverse angles and avoid prior framing patterns."}
- MCQs must include exactly 4 realistic option texts (not "A/B/C/D", not placeholders).
- The correct option text must be in "correct_answer".
- MCQs should include a "difficulty" field among: "easy", "medium", "hard", "expert";
  align MCQ difficulty with the target band "{level}" and order from easier to harder.
- For "code" and "scenario" items, include "correct_answer": null (still include the key) and omit "options".
- For "scenario" items, include:
  - "ideal_answer": concise model answer (3-6 bullets or short paragraph),
  - "rubric": object with 3-5 criteria and weights that sum to 1.0.
- CRITICAL: Generate UNIQUE questions - no duplicates, no recycled content, no repeated questions.
- CRITICAL: Every question must be role-specific to "{role}" - reject any generic or off-topic questions
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
def _synthesize_options(role: str, skills: List[str], question: str, seed_int: int = 0) -> List[str]:
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

    # Pick deterministically using stable hash + seed.
    base = hashlib.sha256(f"{question}|{seed_int}".encode("utf-8")).hexdigest()
    bucket = int(base[:8], 16) % len(bank)
    opts = bank[bucket].copy()

    # Ensure four unique strings
    opts = [str(x) for x in opts][:4]
    while len(opts) < 4:
        opts.append(f"Relevant option {len(opts)+1}")

    return opts[:4]


def _normalize_items(
    items: List[Dict[str, Any]],
    comp: List[Dict[str, Any]],
    candidate: Candidate,
    seed_int: int = 0,
) -> List[Dict[str, Any]]:
    """
    Ensures each item has required keys/types and matches the requested composition order/length.
    Fills in real MCQ options and difficulty if missing.
    """
    normalized: List[Dict[str, Any]] = []

    def _mk_mcq_stub(q: str = "Which option is most appropriate?") -> Dict[str, Any]:
        opts = _synthesize_options(candidate.job_role or "Role", candidate.skills or [], q, seed_int=seed_int)
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
    candidate_level = _level_from_years(_years_from_string(candidate.experience))
    mcq_difficulties = _difficulty_sequence(mcq_count, candidate_level)
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
                options = _synthesize_options(role, skills, qtext, seed_int=seed_int)

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
            if item_type == "scenario":
                ideal = candidate_item.get("ideal_answer")
                rubric = candidate_item.get("rubric")
                if isinstance(ideal, str) and ideal.strip():
                    normalized[-1]["ideal_answer"] = ideal.strip()
                if isinstance(rubric, dict) and rubric:
                    normalized[-1]["rubric"] = rubric

    return normalized


def _fallback_questions(
    candidate: Candidate,
    comp: List[Dict[str, Any]],
    seed_int: int = 0,
) -> List[Dict[str, Any]]:
    """
    Deterministic fallback set in case the API response isn't valid JSON.
    Creates realistic option texts for MCQs (not A/B/C/D).
    """
    role = candidate.job_role or "the role"
    skills = candidate.skills or []
    rng = random.Random(seed_int)

    out: List[Dict[str, Any]] = []
    mcq_seen = 0
    candidate_level = _level_from_years(_years_from_string(candidate.experience))
    mcq_difficulties = _difficulty_sequence(sum(1 for c in comp if c.get("type") == "mcq"), candidate_level)
    type_seen = {"mcq": 0, "code": 0, "scenario": 0}

    for spec in comp:
        t = spec["type"]
        type_seen[t] = type_seen.get(t, 0) + 1
        if t == "mcq":
            topic = rng.choice(skills) if skills else "core responsibilities"
            q_templates = [
                f"As a {role}, which approach is most effective for {topic}?",
                f"In {role} work, which option best improves outcomes related to {topic}?",
                f"Which decision is most appropriate for a {role} handling {topic}?",
                f"Given limited time and resources, what is the best {role} action for {topic}?",
                f"Which trade-off is most defensible for a {role} optimizing {topic}?",
                f"What is the strongest first step for a {role} dealing with {topic} under pressure?",
            ]
            qtext = f"{rng.choice(q_templates)} (Q{type_seen[t]})"
            opts = _synthesize_options(role, skills, qtext, seed_int=seed_int + type_seen[t])
            out.append({
                "type": "mcq",
                "question": qtext,
                "options": opts,
                "correct_answer": opts[0],
                "difficulty": mcq_difficulties[min(mcq_seen, len(mcq_difficulties) - 1)],
            })
            mcq_seen += 1
        elif t == "code":
            focus = rng.choice(skills) if skills else "core logic"
            code_templates = [
                f"Write a function for a {role} workflow that validates and processes {focus}.",
                f"Implement a robust utility related to {role} tasks focusing on {focus}.",
                f"Create a production-ready function for {role} use cases involving {focus}.",
                f"Design an idempotent helper for {role} operations around {focus}.",
                f"Implement a safe parser + validator for {role} inputs tied to {focus}.",
                f"Build a fault-tolerant routine for {role} handling of {focus}.",
            ]
            out.append({
                "type": "code",
                "question": f"{rng.choice(code_templates)} (Q{type_seen[t]})",
                "options": [],
                "correct_answer": None,
            })
        else:
            focus = rng.choice(skills) if skills else "critical responsibilities"
            scenario_templates = [
                f"You are a {role}. A high-impact issue appears around {focus}. Describe your response plan.",
                f"As a {role}, outline how you would handle a complex incident involving {focus}.",
                f"Describe a step-by-step strategy a {role} should take for a risky situation related to {focus}.",
                f"You must brief leadership on a {focus} failure. What concrete recovery plan would you execute as a {role}?",
                f"A cross-team conflict blocks delivery for {focus}. How should a {role} resolve it and reduce recurrence?",
                f"An urgent customer-impacting problem emerges in {focus}. Explain your first-hour and first-day actions as a {role}.",
            ]
            rubric_variants = [
                [
                    {"name": "Problem understanding", "weight": 0.25},
                    {"name": "Actionable plan", "weight": 0.30},
                    {"name": "Risk management", "weight": 0.25},
                    {"name": "Communication clarity", "weight": 0.20},
                ],
                [
                    {"name": "Root cause analysis", "weight": 0.30},
                    {"name": "Execution quality", "weight": 0.30},
                    {"name": "Stakeholder alignment", "weight": 0.20},
                    {"name": "Post-incident learning", "weight": 0.20},
                ],
                [
                    {"name": "Prioritization", "weight": 0.25},
                    {"name": "Technical correctness", "weight": 0.30},
                    {"name": "Reliability safeguards", "weight": 0.25},
                    {"name": "Communication and ownership", "weight": 0.20},
                ],
            ]
            out.append({
                "type": "scenario",
                "question": f"{rng.choice(scenario_templates)} (Q{type_seen[t]})",
                "options": [],
                "correct_answer": None,
                "ideal_answer": (
                    rng.choice([
                        f"Identify risk drivers for {focus}, prioritize impact, communicate stakeholders, execute mitigation steps, validate outcomes, and document lessons learned.",
                        f"Clarify scope and severity for {focus}, define immediate containment, assign owners, communicate timeline, and confirm recovery criteria.",
                        f"Balance short-term stabilization of {focus} with long-term remediation, while documenting decisions, risks, and follow-up controls.",
                    ])
                ),
                "rubric": {
                    "criteria": rng.choice(rubric_variants)
                },
            })
    return out


def _question_fingerprint(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest() if clean else ""


def _shuffle_mcq_options(items: List[Dict[str, Any]], rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    rng = rng or random.SystemRandom()
    out: List[Dict[str, Any]] = []
    for item in items:
        if str(item.get("type", "")).lower() == "mcq":
            options = list(item.get("options") or [])
            if len(options) == 4:
                rng.shuffle(options)
                item = dict(item)
                item["options"] = options
                # correct_answer stays as text (not index), so shuffling is safe.
                if item.get("correct_answer") not in options:
                    item["correct_answer"] = options[0]
        out.append(item)
    return out


def _ensure_unique_questions(
    items: List[Dict[str, Any]],
    comp: List[Dict[str, Any]],
    candidate: Candidate,
    previous_questions: Optional[List[str]],
    seed_int: int = 0,
) -> List[Dict[str, Any]]:
    seen = {_question_fingerprint(q) for q in (previous_questions or []) if q}
    fallback = _fallback_questions(candidate, comp, seed_int=seed_int)
    out: List[Dict[str, Any]] = []

    for i, spec in enumerate(comp):
        desired_type = spec.get("type", "mcq")
        candidate_item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
        if str(candidate_item.get("type", "")).lower() != desired_type:
            candidate_item = fallback[i]

        qtext = str(candidate_item.get("question") or "").strip()
        fp = _question_fingerprint(qtext)
        if (not fp) or (fp in seen):
            candidate_item = fallback[i]
            qtext = str(candidate_item.get("question") or "").strip()
            fp = _question_fingerprint(qtext)
            # As a final guard, force uniqueness deterministically.
            suffix = 2
            while fp and fp in seen:
                candidate_item = dict(candidate_item)
                candidate_item["question"] = f"{qtext} (Variant {suffix})"
                fp = _question_fingerprint(candidate_item["question"])
                suffix += 1

        if fp:
            seen.add(fp)
        out.append(candidate_item)
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

def generate_test(
    candidate: Candidate,
    mcq_count: int = 0,
    scenario_count: int = 0,
    code_count: int = 0,
    question_count: Optional[int] = None,
    previous_questions: Optional[List[str]] = None,
    seed: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Generates a test based on explicit composition parameters provided by the sender.
    Uses candidate experience to tune difficulty band.
    Ensures uniqueness by avoiding previously generated questions.

    Args:
        candidate: Candidate model
        mcq_count: Number of MCQ questions (default 0)
        scenario_count: Number of scenario questions (default 0)
        code_count: Number of coding questions (default 0)
        question_count: Fallback total question count for backward compatibility (default None)
        previous_questions: List of previously generated question texts to avoid duplicates (default None)

    Returns:
        List of normalized question dicts compatible with your evaluator/UI.
    """
    comp = _build_composition(mcq_count, scenario_count, code_count, question_count)
    
    if len(comp) == 0:
        # Fallback: at least one question
        comp = [{"type": "mcq"}]
    
    seed_int = _seed_to_int(seed)
    prompt = _build_prompt(
        candidate,
        comp,
        previous_questions=previous_questions,
        variation_hint=_prompt_variation(seed_int),
    )
    rng = random.Random(seed_int)

    try:
        content = _call_openai(prompt)
        content = _strip_code_fences(content)
        data = json.loads(content)

        if not isinstance(data, list):
            raise ValueError("Model did not return a list")

        normalized = _normalize_items(data, comp, candidate, seed_int=seed_int)
        unique = _ensure_unique_questions(normalized, comp, candidate, previous_questions, seed_int=seed_int)
        shuffled = _shuffle_mcq_options(unique, rng=rng)
        return shuffled

    except Exception:
        # Robust fallback that still respects the requested composition
        fallback = _fallback_questions(candidate, comp, seed_int=seed_int)
        unique = _ensure_unique_questions(fallback, comp, candidate, previous_questions, seed_int=seed_int)
        shuffled = _shuffle_mcq_options(unique, rng=rng)
        return shuffled
