# app/middleware/rate_limit.py
# In-memory rate limit per client IP. Safe defaults: 120 req/min per IP.

import os
import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("smarthirex.ratelimit")

# Default: 120 requests per 60 seconds per IP (configurable via RATE_LIMIT_PER_MINUTE)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
WINDOW_SEC = 60
# In-memory: (ip -> list of timestamps in window). Not distributed; for single-instance.
_store: defaultdict = defaultdict(list)
_store_ts = 0


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _is_over_limit(ip: str) -> bool:
    global _store_ts
    now = time.monotonic()
    # Prune old entries periodically
    if now - _store_ts > 30:
        _store_ts = now
        cutoff = now - WINDOW_SEC
        for k in list(_store.keys()):
            _store[k] = [t for t in _store[k] if t > cutoff]
    times = _store[ip]
    cutoff = now - WINDOW_SEC
    times[:] = [t for t in times if t > cutoff]
    if len(times) >= RATE_LIMIT_PER_MINUTE:
        return True
    times.append(now)
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = _get_client_ip(request)
        if _is_over_limit(ip):
            logger.warning("rate_limit_exceeded", extra={"ip": ip})
            return JSONResponse(
                {"detail": "Too many requests. Please try again later."},
                status_code=429,
            )
        return await call_next(request)
