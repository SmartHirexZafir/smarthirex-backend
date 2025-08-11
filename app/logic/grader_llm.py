# app/logic/grader_llm.py
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

# --- SDK compatibility: prefer new OpenAI v1, fall back to legacy ---
_NEW_SDK = False
try:
    from openai import OpenAI  # type: ignore
    _NEW_SDK = True
except Exception:
    _NEW_SDK = False

# keep legacy import available even if unused
try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return 0.0


def _json_coerce(s: str) -> Dict[str, Any]:
    """
    Best-effort to parse JSON returned by the model.
    - strips code fences
    - extracts first {...} block if needed
    """
    s = (s or "").strip()
    if not s:
        return {}
    # strip ```json ... ``` fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()

    # try direct json
    try:
        return json.loads(s)
    except Exception:
        pass

    # try to find JSON object substring
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _get_api_key_and_model() -> Tuple[str, str]:
    """
    Model picked via env:
      OPENAI_API_KEY    -> required
      OPENAI_MODEL      -> optional; defaults: gpt-4o-mini (new SDK) or gpt-4 (legacy)
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if _NEW_SDK else os.getenv("OPENAI_MODEL", "gpt-4")
    return api_key, model


def _chat_once(prompt: str, temperature: float = 0.0, max_retries: int = 2, timeout_backoff: float = 0.8) -> str:
    """
    Calls the model and returns message content (string).
    Retries lightly on transient errors.
    """
    api_key, model = _get_api_key_and_model()
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if _NEW_SDK:
                client = OpenAI(api_key=api_key)  # type: ignore
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise grading assistant. Always respond with strict JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            else:
                if openai is None:
                    raise RuntimeError("openai library not available")
                openai.api_key = api_key  # type: ignore
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise grading assistant. Always respond with strict JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return resp["choices"][0]["message"]["content"]  # type: ignore[index]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(timeout_backoff * (attempt + 1))
                continue
            break
    # bubble the final error (caller will catch and fall back)
    raise last_err if last_err else RuntimeError("Unknown LLM error")


def _mk_scenario_prompt(answer: str, question: str, rubric: Optional[Dict[str, Any]]) -> str:
    """
    Build a strict-JSON grading prompt for scenario/short-answer questions.
    """
    keywords = []
    notes = ""
    if rubric:
        if isinstance(rubric.get("keywords"), list):
            keywords = [str(k) for k in rubric.get("keywords") if str(k).strip()]
        if isinstance(rubric.get("notes"), str):
            notes = rubric.get("notes", "")

    return f"""
Grade the candidate's scenario/short-answer response. Evaluate coverage of the core logic/concepts.
Focus on whether the answer addresses key points, trade-offs, and risks. Penalize generic fluff.

QUESTION:
{question.strip()}

CANDIDATE_ANSWER:
{answer.strip()}

RUBRIC_KEYWORDS (optional, weight them higher if present):
{keywords}

RUBRIC_NOTES (optional):
{notes}

Return STRICT JSON ONLY in this schema:
{{
  "score": 0.0_to_1.0_number,
  "explanation": "short reason (<= 60 words)"
}}
""".strip()


def _mk_code_prompt(answer: str, question: str, rubric: Optional[Dict[str, Any]]) -> str:
    """
    Build a strict-JSON grading prompt for coding questions.
    The grader checks for core algorithm/logic, correctness, complexity, and clarity.
    """
    keywords = []
    notes = ""
    if rubric:
        if isinstance(rubric.get("keywords"), list):
            keywords = [str(k) for k in rubric.get("keywords") if str(k).strip()]
        if isinstance(rubric.get("notes"), str):
            notes = rubric.get("notes", "")

    return f"""
Grade the candidate's coding solution. Evaluate correctness of core logic and algorithmic approach.
If exact execution isn't provided, infer from pseudocode/description. Consider:
- coverage of required steps / edge cases
- time/space complexity when relevant
- clarity and structure (function decomposition, naming)
- use the rubric keywords if supplied

QUESTION:
{question.strip()}

CANDIDATE_CODE_OR_DESCRIPTION:
{answer.strip()}

RUBRIC_KEYWORDS (optional, weight them higher if present):
{keywords}

RUBRIC_NOTES (optional):
{notes}

Return STRICT JSON ONLY in this schema:
{{
  "score": 0.0_to_1.0_number,
  "explanation": "short reason (<= 60 words)"
}}
""".strip()


def _grade_with_llm(prompt: str) -> Dict[str, Any]:
    """
    Core LLM grading: returns {"score": float in [0,1], "explanation": str}
    Raises on API issues; caller should catch and fallback.
    """
    raw = _chat_once(prompt, temperature=0.0)
    data = _json_coerce(raw)

    score = _clamp(data.get("score", 0.0))
    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        explanation = "Graded by LLM."
    return {"score": score, "explanation": explanation}


# -------- Public functions (kept) -------------------------------------------

def grade_scenario_llm(
    *,
    answer: str,
    question: str,
    rubric: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-based grader for scenario/short-answer items.

    Returns:
      {
        "score": float in [0,1],
        "explanation": "short reason"
      }

    Raises:
      Exception if the LLM call fails (recommended to catch and use rules-based fallback).
    """
    prompt = _mk_scenario_prompt(answer=answer or "", question=question or "", rubric=rubric or {})
    return _grade_with_llm(prompt)


def grade_code_llm(
    *,
    answer: str,
    question: str,
    rubric: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM-based grader for coding items.

    Returns:
      {
        "score": float in [0,1],
        "explanation": "short reason"
      }

    Raises:
      Exception if the LLM call fails (recommended to catch and use rules-based fallback).
    """
    prompt = _mk_code_prompt(answer=answer or "", question=question or "", rubric=rubric or {})
    return _grade_with_llm(prompt)


# -------- Wrapper functions (NEW): compatible with evaluator/tests_router ----

def grade_free_form_llm(
    *,
    question: str,
    answer: str,
    max_points: int = 5,
    reference: Optional[str] = None,   # accepted for API compatibility; not required
    q_type: str = "scenario",
    rubric: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper expected by other modules.
    Maps the 0..1 LLM score to points out of `max_points` and returns a rich dict:

    {
      "points": float,
      "max_points": int,
      "feedback": str,
      "is_correct": bool,              # derived using a 0.6 pass threshold
      "confidence": float,             # same as score in [0,1]
      # optional: "rubric": {...}      # left empty unless you supply one
    }
    """
    try:
        if (q_type or "").lower().strip() == "code":
            base = grade_code_llm(answer=answer or "", question=question or "", rubric=rubric)
        else:
            base = grade_scenario_llm(answer=answer or "", question=question or "", rubric=rubric)
    except Exception as e:
        # Bubble up — caller (router/evaluator) already catches & falls back gracefully
        raise

    score = float(base.get("score", 0.0) or 0.0)
    expl = str(base.get("explanation", "") or "Graded by LLM.").strip()
    points = round(max(0.0, min(1.0, score)) * float(max_points), 2)

    out: Dict[str, Any] = {
        "points": points,
        "max_points": int(max_points),
        "feedback": expl,
        "is_correct": bool(score >= 0.6),
        "confidence": _clamp(score),
    }
    # If you pass a structured rubric and want it echoed back for UI, include it.
    # (Your tests_router looks for .get("rubric") optionally.)
    if rubric and isinstance(rubric, dict) and rubric:
        out["rubric"] = rubric
    return out


# Alias used by some imports
def grade_freeform_llm(**kwargs):
    return grade_free_form_llm(**kwargs)
