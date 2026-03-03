"""
Pakistan Standard Time (Asia/Karachi, UTC+5) for application-wide datetime handling.
All user-facing and business logic times use this timezone.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

PK_TZ = ZoneInfo("Asia/Karachi")


def now_pk() -> datetime:
    """Return current time as timezone-aware datetime in Pakistan (Asia/Karachi)."""
    return datetime.now(PK_TZ)


def parse_datetime_pk(dt_input: Optional[str | datetime]) -> Optional[datetime]:
    """
    Parse to timezone-aware datetime in Asia/Karachi.
    Does NOT convert: the wall-clock time (hour, minute, etc.) is kept as-is and
    assigned to Pakistan. So 01:15 stays 01:15 in the database.
    - Naive datetimes: treat as Pakistan time.
    - String with Z or any offset: use the numeric date/time as Pakistan time (no UTC conversion).
    """
    if not dt_input:
        return None

    if isinstance(dt_input, datetime):
        # Use same wall-clock, assign Pakistan
        return datetime(
            dt_input.year, dt_input.month, dt_input.day,
            dt_input.hour, dt_input.minute, dt_input.second,
            dt_input.microsecond, tzinfo=PK_TZ,
        )

    try:
        raw = str(dt_input).strip()
        if raw.endswith("Z"):
            raw = raw[:-1]  # parse as naive so 01:15 stays 01:15
        dt = datetime.fromisoformat(raw)
        # Keep wall-clock exactly; assign Pakistan (no conversion)
        return datetime(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second,
            getattr(dt, "microsecond", 0),
            tzinfo=PK_TZ,
        )
    except (ValueError, AttributeError, TypeError):
        return None
