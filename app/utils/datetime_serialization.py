from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def serialize_utc(dt: datetime) -> str:
    """
    Serialize datetime as strict UTC ISO-8601 with trailing Z.
    - Naive datetimes are assumed UTC.
    - Always includes seconds.
    - Never returns +00:00 offset.
    """
    if not isinstance(dt, datetime):
        raise TypeError("serialize_utc expects datetime")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def serialize_utc_any(value: Any) -> Optional[str]:
    """
    Best-effort serializer for datetime-like values.
    Returns None if value is empty/invalid.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return serialize_utc(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        # Accept trailing Z
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return serialize_utc(dt)
    except Exception:
        return None
