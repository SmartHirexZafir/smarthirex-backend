# app/middleware/request_logging.py
# Log 4xx/5xx responses as structured JSON. No PII in logs.

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("smarthirex.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        status = response.status_code
        if status >= 400:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warning(
                "4xx_5xx_response",
                extra={
                    "status": status,
                    "path": request.url.path,
                    "method": request.method,
                    "latency_ms": latency_ms,
                },
            )
        return response
