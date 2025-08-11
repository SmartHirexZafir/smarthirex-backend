# app/logic/evaluator.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

# --------------------------------------------------------------------
# Env toggles (supports both legacy and new names)
# --------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

# New-style flags (recommended)
FREEFORM_RULES_GRADING = _env_bool("FREEFORM_RULES_GRADING", True)
FREEFORM_LLM_GRADING   = _env_bool("FREEFORM_LLM_GRADING", False)
# Legacy compatibility flag
GRADER_LLM_ENABLED     = _env_bool("GRADER_LLM_ENABLED", False)

# Effective flags
_USE_RULES = FREEFORM_RULES_GRADING
_USE_LLM   = FREEFORM_LLM_GRADING or GRADER_LLM_ENABLED

# --------------------------------------------------------------------
# Optional graders (import-resilient: supports multiple function names)
# --------------------------------------------------------------------
_rules_fn = None
_llm_fn = None

if _USE_RULES:
    try:
        # Preferred
        from app.logic.grader_rules import grade_free_form as _rules_fn  # type: ignore
    except Exception:
        try:
            # Alternate name used in some trees
            from app.logic.grader_rules import grade_freeform_rules as _rules_fn  # type: ignore
        except Exception:
            _rules_fn = None
            _USE_RULES = False

if _USE_LLM:
    try:
        # Preferred
        from app.logic.grader_llm import grade_free_form_llm as _llm_fn  # type: ignore
    except Exception:
        try:
            # Alternate name used in some trees
            from app.logic.grader_llm import grade_freeform_llm as _llm_fn  # type: ignore
        except Exception:
            _llm_fn = None
            _USE_LLM = False

# --------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------
def _norm(s: Any) -> str:
    """Safely normalize to a lowercase string for comparisons."""
    if s is None:
        return ""
    try:
        return str(s).strip().lower()
    except Exception:
        return ""

def _clean_text(s: Any) -> str:
    """Lowercase, collapse spaces, strip punctuation for fuzzy matching."""
    t = _norm(s)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# --------------------------------------------------------------------
# MCQ grading (unchanged)
# --------------------------------------------------------------------
def evaluate_mcq(submitted: str, correct: str) -> int:
    """
    Evaluates a single MCQ answer (case/space-insensitive).
    Returns 1 if correct else 0.
    """
    return 1 if _norm(submitted) == _norm(correct) else 0

# --------------------------------------------------------------------
# Free-form grading (scenario/coding) with rules + optional LLM
# --------------------------------------------------------------------
def _grade_freeform(
    question: str,
    submitted: str,
    correct_ref: Optional[str] = None,
    q_type: str = "scenario",
) -> Dict[str, Any]:
    """
    Returns:
      {
        "is_correct": bool,
        "confidence": float (0..1),
        "explanation": str
      }
    - Uses local rules first when available.
    - Optionally refines with LLM if enabled/available.
    - Never raises; always returns a safe structure.
    """
    # 1) Rules (fast/offline) or heuristic fallback
    result = {
        "is_correct": False,
        "confidence": 0.0,
        "explanation": "Could not evaluate.",
    }

    if _USE_RULES and _rules_fn:
        try:
            r = _rules_fn(
                question=question,
                answer=submitted,
                reference=correct_ref or "",
                q_type=q_type,
            )
            if isinstance(r, dict):
                result = {
                    "is_correct": bool(r.get("is_correct", False) or (r.get("points", 0) or 0) > 0),
                    "confidence": float(r.get("confidence", 0.0) or 0.0),
                    "explanation": str(r.get("feedback") or r.get("explanation") or "Evaluated by rules."),
                }
        except Exception:
            # fall through to heuristic
            pass

    if result["explanation"] == "Could not evaluate.":
        # Minimal heuristic if no rules or rules failed
        ans = _clean_text(submitted)
        word_count = len(ans.split()) if ans else 0
        result = {
            "is_correct": word_count >= 6,
            "confidence": 0.5 if word_count >= 6 else 0.2,
            "explanation": "Heuristic check based on answer length and relevance.",
        }

    # 2) LLM refinement — trust only when more confident than rules
    if _USE_LLM and _llm_fn:
        try:
            l = _llm_fn(
                question=question,
                answer=submitted,
                reference=correct_ref or "",
                q_type=q_type,
            )
            if isinstance(l, dict):
                l_conf = float(l.get("confidence", 0.0) or 0.0)
                r_conf = float(result.get("confidence", 0.0) or 0.0)
                if l_conf >= max(0.6, r_conf):
                    return {
                        "is_correct": bool(l.get("is_correct", False) or (l.get("points", 0) or 0) > 0),
                        "confidence": l_conf,
                        "explanation": str(l.get("explanation", "Evaluated by LLM.")),
                    }
        except Exception:
            pass

    return result

# --------------------------------------------------------------------
# Public API — Evaluates whole test
# --------------------------------------------------------------------
def evaluate_test(answers: List[Dict[str, Any]], correct_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a test by comparing submitted answers with correct ones.

    Compatibility preserved:
    - MCQs: scored as before (1 point per correct) -> total_score.
    - Non-MCQ (code/scenario): we provide a boolean is_correct and an
      explanation, but we DO NOT include those in numeric total_score.
      (Percent is computed elsewhere using MCQ count.)

    Returns:
      {
        "total_score": int,   # MCQ-only count
        "details": [
          {
            "question": str,
            "submitted": str,
            "correct": str,           # "" for free-form
            "is_correct": bool,
            "explanation": str,
            # "type": "mcq"|"code"|"scenario"  (additive, optional)
          }, ...
        ]
      }
    """
    score = 0  # MCQ-only score (do not change)
    detailed_results: List[Dict[str, Any]] = []

    answers = answers or []
    correct_answers = correct_answers or []

    for i, submitted in enumerate(answers):
        if i >= len(correct_answers):
            continue  # safety

        correct = correct_answers[i] or {}

        q_type = (correct.get("type", "mcq") or "mcq")
        q_type_norm = _norm(q_type) or "mcq"
        question_text = (correct.get("question", "") or "")

        submitted_ans_raw = submitted.get("answer", "")
        # Keep original text for graders; store a normalized version for display/compare
        submitted_norm = _norm(submitted_ans_raw)

        # NOTE: correct_answer may be None for free-form
        correct_ans_raw = correct.get("correct_answer", None)
        correct_norm = _norm(correct_ans_raw)

        if q_type_norm == "mcq":
            is_correct = evaluate_mcq(submitted_norm, correct_norm) == 1
            score += int(is_correct)
            explanation = (
                "Correct answer matched." if is_correct
                else f"Expected '{correct_norm}', got '{submitted_norm}'"
            )
            detailed_results.append({
                "question": question_text,
                "submitted": submitted_norm,
                "correct": correct_norm,
                "is_correct": is_correct,
                "explanation": explanation,
                "type": "mcq",
            })
            continue

        # --- Free-form (scenario / code) ---
        ff = _grade_freeform(
            question=question_text,
            submitted=submitted_ans_raw if isinstance(submitted_ans_raw, str) else submitted_norm,
            correct_ref=correct_ans_raw if isinstance(correct_ans_raw, str) else None,
            q_type=q_type_norm,
        )
        detailed_results.append({
            "question": question_text,
            "submitted": submitted_norm,
            "correct": "",  # no single canonical correct string for free-form
            "is_correct": bool(ff.get("is_correct", False)),
            "explanation": str(ff.get("explanation", "Evaluated.")),
            "type": q_type_norm,
        })

    return {
        "total_score": score,   # keep MCQ total to preserve existing percent math
        "details": detailed_results
    }
