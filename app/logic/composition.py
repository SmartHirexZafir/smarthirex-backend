# app/logic/composition.py
from __future__ import annotations

import math
import re
from typing import List, Dict

__all__ = [
    "years_from_string",
    "composition_for_experience",
]

# ------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------

def years_from_string(experience: str | None) -> int:
    """
    Extract the first integer from an experience string.
    Examples:
      "2 years" -> 2
      "3-4 yrs" -> 3
      "7+ years" -> 7
      "" or None -> 0
    """
    if not experience:
        return 0
    m = re.search(r"(\d+)", str(experience))
    return int(m.group(1)) if m else 0


def composition_for_experience(
    *,
    years: int,
    question_count: int,
) -> List[Dict[str, str]]:
    """
    DEPRECATED: This function is kept for backward compatibility only.
    Experience-based logic has been removed. The sender now decides question composition.
    
    This function now returns all MCQs as a fallback for backward compatibility.
    New code should use explicit composition parameters instead.
    """
    question_count = max(1, int(question_count))
    # Return all MCQs as fallback (no experience-based logic)
    return [{"type": "mcq"}] * question_count


# ------------------------------------------------------------
# Internal utilities
# ------------------------------------------------------------

def _distribute(
    total: int,
    weights: dict[str, float],
    mins: dict[str, int],
) -> dict[str, int]:
    """
    Turn weights + minimums into integer counts that sum to `total`.

    Strategy:
      1) Start with mins.
      2) Spread the remainder by weights using largest-remainder (Hamilton) rounding.
      3) Ensure no negative allocations and that sum matches total.
    """
    keys = ("mcq", "code", "scenario")

    # 1) start at minimums
    alloc = {k: max(0, int(mins.get(k, 0))) for k in keys}
    used = sum(alloc.values())
    remaining = total - used
    if remaining <= 0:
        # Trim down if mins exceed total (keep priority order: mcq, code, scenario)
        return _trim_to_total(alloc, total, order=keys)

    # 2) proportional shares on the remainder
    weight_sum = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys)
    if weight_sum == 0:
        # No weights? dump remainder into MCQs by default
        alloc["mcq"] += remaining
        return alloc

    # Raw fractional shares
    shares = {k: (max(0.0, float(weights.get(k, 0.0))) / weight_sum) * remaining for k in keys}
    floors = {k: int(math.floor(shares[k])) for k in keys}
    rema   = {k: shares[k] - floors[k] for k in keys}

    # Base with floors
    for k in keys:
        alloc[k] += floors[k]

    # Distribute leftover by largest fractional remainder
    left = remaining - sum(floors.values())
    if left > 0:
        for k in sorted(keys, key=lambda x: rema[x], reverse=True):
            if left == 0:
                break
            alloc[k] += 1
            left -= 1

    # Final sanity pass
    return _trim_to_total(alloc, total, order=keys)


def _trim_to_total(alloc: dict[str, int], total: int, order: tuple[str, ...]) -> dict[str, int]:
    """
    Ensure non-negative counts and exact sum == total.
    If we need to trim, drop from the end of `order` first (scenario -> code -> mcq for typical calls,
    but caller supplies the order they want to preserve).
    """
    for k in alloc:
        alloc[k] = max(0, int(alloc[k]))

    current = sum(alloc.values())
    if current == total:
        return alloc

    if current > total:
        # Reduce starting from the last types in the provided order
        over = current - total
        for k in reversed(order):
            take = min(over, alloc[k])
            alloc[k] -= take
            over -= take
            if over == 0:
                break
    else:
        # Add the deficit to the first type in order (usually MCQ)
        alloc[order[0]] += (total - current)

    return alloc
