# app/logic/grader_rules.py
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Set, Tuple


"""
Rules-based fallback grader for free-form answers (scenario & coding).

Public functions:
  - grade_scenario_rules(answer: str, question: str, rubric: dict | None) -> dict
  - grade_code_rules(answer: str, question: str, rubric: dict | None) -> dict

Return shape (normalized):
  {
    "score": float in [0,1],
    "explanation": "short reason (<= 60 words)"
  }

Heuristic model (deterministic):
  1) Keyword coverage (driven by rubric.keywords if provided; otherwise
     auto-derives crude candidates from the question text).
  2) Quality signals (structure, clarity; code signals for coding).
  3) Length/coherence bonus (too short penalized, wildly long damped).
"""


# ----------------- Text utils -----------------

_WORD_RE = re.compile(r"[A-Za-z0-9_#+\-\/\.]{2,}")

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def _words(s: str) -> List[str]:
    return _WORD_RE.findall(_normalize(s))

def _unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    try:
        return max(0.0, min(1.0, float(num) / float(den)))
    except Exception:
        return 0.0


# ----------------- Keyword extraction/coverage -----------------

# Words we tend to ignore when auto-pulling "keywords" from the question
_STOP = set("""
the a an and or but with without within into onto from to of in on for by about as at be being been is are was were
this that these those which who whom whose where when why how what
you your yours we our ours they their theirs he his she her it its
""".split())

# Booster map for common techy tokens that often matter in grading
_TECH_BOOST = {
    "time", "space", "complexity", "optimize", "optimize", "scale", "scalable",
    "latency", "throughput",
    "edge", "case", "edgecase", "null", "none", "nan",
    "test", "unit", "integration", "validate", "validation", "verify",
    "secure", "security", "auth", "oauth", "jwt", "encrypt",
    "index", "cache", "caching",
    "sql", "nosql", "join", "transaction", "isolation",
    "api", "rest", "grpc", "queue", "pubsub",
    "thread", "lock", "race", "deadlock", "concurrency", "parallel",
}

def _derive_keywords_from_question(question: str, max_k: int = 8) -> List[str]:
    tokens = [w for w in _words(question) if w not in _STOP]
    # prefer "tech-like" tokens, then lengthier nouns-ish tokens
    boosted = sorted(tokens, key=lambda t: (t in _TECH_BOOST, len(t)), reverse=True)
    cand = _unique(boosted)[: max_k * 2]
    # keep only reasonably informative tokens
    cand = [c for c in cand if len(c) >= 3]
    # cap
    return cand[:max_k]

def _keyword_coverage(answer: str, keywords: List[str]) -> Tuple[float, List[str]]:
    if not keywords:
        return (0.0, [])
    a_words = set(_words(answer))
    kws = [k.lower() for k in keywords if k and isinstance(k, str)]
    kws = _unique(kws)
    matched = [k for k in kws if k in a_words]
    return _safe_ratio(len(matched), len(kws)), matched


# ----------------- Quality signals -----------------

# Scenario/short-answer signals: process & structure
_SCENARIO_SIGNALS = {
    # step-by-step/process markers
    "analyze", "investigate", "assess", "gather", "baseline",
    "design", "plan", "propose", "prioritize", "tradeoff", "trade-offs",
    "implement", "iterate", "monitor", "track", "measure", "kpi",
    "risk", "mitigate", "fallback", "rollback",
    "stakeholder", "communication", "timeline",
    # craftsmanship markers
    "edge", "case", "validate", "verify", "document",
}

# Coding signals: structure & correctness hints
_CODE_SIGNALS = {
    "def", "function", "class", "method", "return",
    "if", "else", "elif", "switch", "case",
    "for", "while", "recursion", "recursive",
    "try", "except", "catch", "finally",
    "test", "unit", "assert",
    "null", "none", "nil",
    "complexity", "o(", "optimize",
    "edge", "case", "validate",
}

def _count_signals(answer: str, signals: Set[str]) -> int:
    a = _normalize(answer)
    total = 0
    for s in signals:
        if s in a:
            total += 1
    return total

def _length_score(answer: str, min_words: int = 25, max_words: int = 250) -> float:
    n = len(_words(answer))
    if n <= 5:
        return 0.0
    if n < min_words:
        # ramp up to 0.6 as user approaches min_words
        return 0.2 + 0.4 * _safe_ratio(n, min_words)
    # within range gets strong credit; beyond is damped
    if n <= max_words:
        return 0.9
    # decay slowly after max_words
    overflow = n - max_words
    return max(0.5, 0.9 - min(0.4, overflow / 600.0))  # gentle decay


# ----------------- Scoring core -----------------

def _bounded_explanation(text: str, max_words: int = 60) -> str:
    toks = _words(text)
    if len(toks) <= max_words:
        return text.strip()
    trimmed = " ".join(toks[:max_words])
    return trimmed + "…"

def _grade_freeform(
    *,
    answer: str,
    question: str,
    rubric: Dict[str, Any] | None,
    mode: str,  # "scenario" | "code"
) -> Dict[str, Any]:
    # Pull rubric keywords if provided; else derive from the question
    rubric = rubric or {}
    rubric_keywords = rubric.get("keywords") if isinstance(rubric.get("keywords"), list) else []
    if not rubric_keywords:
        rubric_keywords = _derive_keywords_from_question(question)

    # (1) Keyword coverage (major weight)
    cov, matched = _keyword_coverage(answer, rubric_keywords)

    # (2) Quality signals (mode-specific)
    if mode == "code":
        signal_count = _count_signals(answer, _CODE_SIGNALS)
        signal_norm = _safe_ratio(signal_count, 10)  # cap normalization
    else:
        signal_count = _count_signals(answer, _SCENARIO_SIGNALS)
        signal_norm = _safe_ratio(signal_count, 8)

    # (3) Length/coherence bonus
    len_score = _length_score(answer)

    # Combine with weights
    # Rationale:
    # - Keywords show topical coverage (0.55)
    # - Structure/quality signals add (0.30)
    # - Length/coherence guard adds (0.15)
    w_cov, w_sig, w_len = 0.55, 0.30, 0.15
    raw = w_cov * cov + w_sig * signal_norm + w_len * len_score

    # Optional rubric bump if "notes" present and partially matched
    notes = rubric.get("notes")
    if isinstance(notes, str) and notes.strip():
        # crude overlap ratio between notes tokens and answer tokens
        ov = _keyword_coverage(answer, _unique(_words(notes)))[0]
        raw = min(1.0, raw + 0.10 * ov)

    score = max(0.0, min(1.0, float(raw)))

    # Build short explanation
    reason_bits: List[str] = []
    if matched:
        reason_bits.append(f"covered {len(matched)}/{len(rubric_keywords)} key points ({', '.join(matched[:4])}{'…' if len(matched)>4 else ''})")
    else:
        reason_bits.append("limited coverage of key points")
    if signal_count > 0:
        reason_bits.append(f"{signal_count} quality signals detected")
    # hint about brevity or verbosity
    wc = len(_words(answer))
    if wc < 20:
        reason_bits.append("very brief response")
    elif wc > 300:
        reason_bits.append("overly long; partial credit given")

    expl = _bounded_explanation("; ".join(reason_bits))

    return {"score": score, "explanation": expl}


# ----------------- Public API -----------------

def grade_scenario_rules(
    *,
    answer: str,
    question: str,
    rubric: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Rules-based grader for scenario/short-answer responses.
    """
    return _grade_freeform(answer=answer or "", question=question or "", rubric=rubric, mode="scenario")


def grade_code_rules(
    *,
    answer: str,
    question: str,
    rubric: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Rules-based grader for coding responses (code or pseudocode).
    """
    return _grade_freeform(answer=answer or "", question=question or "", rubric=rubric, mode="code")
