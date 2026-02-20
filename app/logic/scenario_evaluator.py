# app/logic/scenario_evaluator.py
"""
GPT-based evaluator for scenario questions with intelligent leniency system.
No keyword matching, no strict word detection - pure AI-driven evaluation.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, Optional

# --- SDK compatibility: prefer new OpenAI v1, fall back to legacy ---
_NEW_SDK = False
try:
    from openai import OpenAI  # type: ignore
    _NEW_SDK = True
except Exception:
    _NEW_SDK = False

try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


def _get_api_key_and_model() -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if _NEW_SDK else os.getenv("OPENAI_MODEL", "gpt-4")
    return api_key, model


def _call_openai(prompt: str, temperature: float = 0.3) -> str:
    """
    Calls OpenAI API and returns the response content.
    Uses slightly higher temperature (0.3) for more nuanced evaluation.
    """
    api_key, model = _get_api_key_and_model()
    
    if _NEW_SDK:
        client = OpenAI(api_key=api_key)  # type: ignore
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent, fair, and context-aware test evaluator. Evaluate answers based on conceptual accuracy and understanding, not exact wording. Be lenient but accurate."
                },
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
                {
                    "role": "system",
                    "content": "You are an intelligent, fair, and context-aware test evaluator. Evaluate answers based on conceptual accuracy and understanding, not exact wording. Be lenient but accurate."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]  # type: ignore[index]


def evaluate_scenario_with_leniency(
    question: str,
    candidate_answer: str,
    max_score: float = 1.0,
    ideal_answer: Optional[str] = None,
    rubric: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluates a scenario question using GPT with intelligent leniency.
    
    Leniency rules:
    - ~80% correct → full marks (1.0)
    - Partial/incomplete → proportional scoring (0.4-0.8)
    - Irrelevant/incorrect → lower score (0.0-0.3)
    
    Args:
        question: The scenario question text
        candidate_answer: The candidate's submitted answer
        max_score: Maximum score (default 1.0)
    
    Returns:
        {
            "score": float (0.0 to max_score),
            "normalized_score": float (0.0 to 1.0),
            "explanation": str,
            "is_correct": bool,
            "confidence": float (0.0 to 1.0)
        }
    """
    ideal_block = ""
    if ideal_answer and ideal_answer.strip():
        ideal_block = f"""
REFERENCE IDEAL ANSWER:
{ideal_answer.strip()}
"""

    rubric_block = ""
    if isinstance(rubric, dict) and rubric:
        rubric_block = f"""
RUBRIC:
{json.dumps(rubric, ensure_ascii=True)}
"""

    prompt = f"""Evaluate this scenario question answer intelligently and fairly.

QUESTION:
{question.strip()}

CANDIDATE'S ANSWER:
{candidate_answer.strip()}
{ideal_block}
{rubric_block}

EVALUATION CRITERIA:
1. Conceptual Accuracy: Does the answer demonstrate understanding of the core concepts?
2. Relevance: Is the answer relevant to the question asked?
3. Completeness: Does it address the key points (even if not all details)?
4. Context Awareness: Does it show practical understanding?

LENIENCY RULES (apply these fairly):
- If the answer is ~80% correct or demonstrates strong understanding → give FULL MARKS (1.0)
- If the answer is partially correct or incomplete but shows some understanding → give PROPORTIONAL SCORE (0.4 to 0.8)
- If the answer is irrelevant, incorrect, or shows no understanding → give LOWER SCORE (0.0 to 0.3)

IMPORTANT:
- DO NOT use keyword matching or exact word detection
- Evaluate based on MEANING and CONCEPTUAL ACCURACY
- Be flexible with wording - different phrasings of the same concept should score equally
- Consider the context and intent, not just literal text matching
- If ideal answer/rubric is provided, align scoring against that quality bar.

Return STRICT JSON ONLY in this exact format:
{{
  "score": 0.0_to_1.0_number,
  "explanation": "Brief explanation of the evaluation (2-3 sentences max)",
  "confidence": 0.0_to_1.0_number
}}

The score should be a float between 0.0 and 1.0 representing the quality of the answer.
Apply leniency: if the answer is approximately 80% correct, give it a score of 1.0 (full marks).
"""

    try:
        raw_response = _call_openai(prompt, temperature=0.3)
        raw_response = _strip_code_fences(raw_response)
        
        # Try to parse JSON
        data = json.loads(raw_response)
        
        # Extract and validate score
        raw_score = float(data.get("score", 0.0))
        normalized_score = max(0.0, min(1.0, raw_score))  # Clamp to [0, 1]
        
        # Apply leniency: if score is >= 0.8, round up to 1.0
        if normalized_score >= 0.8:
            normalized_score = 1.0
        
        actual_score = normalized_score * max_score
        explanation = str(data.get("explanation", "Evaluated by AI")).strip()
        confidence = float(data.get("confidence", normalized_score))
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "score": round(actual_score, 2),
            "normalized_score": normalized_score,
            "explanation": explanation if explanation else "Evaluated by AI based on conceptual accuracy.",
            "is_correct": normalized_score >= 0.8,  # Consider >= 80% as correct
            "confidence": confidence,
        }
    
    except Exception as e:
        # Fallback: conservative scoring
        return {
            "score": 0.0,
            "normalized_score": 0.0,
            "explanation": f"Evaluation error: {str(e)[:100]}",
            "is_correct": False,
            "confidence": 0.0,
        }

