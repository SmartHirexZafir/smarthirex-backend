# app/routers/events_router.py
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Try to import your in-memory SSE broker
# (see: app/utils/sse_bus.py)
try:
    from app.utils.sse_bus import broker  # type: ignore
except Exception:  # pragma: no cover
    broker = None  # If unavailable, we will return 501

router = APIRouter(prefix="/events", tags=["events"])


def _sse_headers() -> dict[str, str]:
    """
    Headers that help SSE work reliably through proxies (NGINX, Cloudflare, etc.).
    """
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Disable proxy buffering for NGINX
        "X-Accel-Buffering": "no",
    }


@router.get("/subscribe")
async def subscribe_query(
    channel: str = Query(..., description="Channel name, e.g. 'public' or 'user:<id>'"),
    email: Optional[str] = Query(None, description="Optional organizer email (not required)"),
):
    """
    Server-Sent Events (SSE) subscription endpoint using a *query parameter*.

    The frontend (NotificationsProvider) calls:
      GET /events/subscribe?channel=<channel>[&email=<email>]

    Notes:
    - `broker.subscribe(channel)` yields SSE-formatted strings ("event:", "data:", etc.)
    - Response MUST have media_type="text/event-stream"
    - This is an *in-process* broker; if you run multiple workers,
      each worker maintains its own subscriber set.
    """
    if broker is None:
        raise HTTPException(status_code=501, detail="SSE broker not available")

    # Best-effort: trim/normalize empty channel
    ch = (channel or "").strip()
    if not ch:
        raise HTTPException(status_code=400, detail="channel is required")

    async def event_generator():
        # The broker handles initial comment, heartbeats, and backpressure
        async for msg in broker.subscribe(ch):
            # msg is already SSE-formatted ("event:", "data:", etc.)
            yield msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@router.get("/subscribe/{channel}")
async def subscribe_path(
    channel: str,
    email: Optional[str] = Query(None, description="Optional organizer email (not required)"),
):
    """
    Same SSE subscription, but with *path param* style:
      GET /events/subscribe/<channel>[?email=...]
    """
    if broker is None:
        raise HTTPException(status_code=501, detail="SSE broker not available")

    ch = (channel or "").strip()
    if not ch:
        raise HTTPException(status_code=400, detail="channel is required")

    async def event_generator():
        async for msg in broker.subscribe(ch):
            yield msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ─────────────────────────────────────────────────────────────
# Optional helpers (useful during development / testing)
# ─────────────────────────────────────────────────────────────

class PublishPayload(BaseModel):
    channel: str = Field(..., description="Target channel (e.g. 'public' or 'user:<id>')")
    data: Any = Field(..., description="Arbitrary JSON-serializable payload")
    event: Optional[str] = Field(
        default=None,
        description="Optional SSE event name (e.g. 'meeting.started'). If not provided, a default 'message' event is used.",
    )
    retry: Optional[int] = Field(
        default=None,
        description="Optional SSE 'retry' backoff in ms sent to client.",
    )
    id: Optional[str] = Field(
        default=None,
        description="Optional SSE 'id' field.",
    )


@router.post("/publish")
async def publish_event(
    body: PublishPayload,
    token: Optional[str] = Query(
        default=None,
        description="Optional publish token. If EVENTS_PUBLISH_TOKEN is set in env, this must match.",
    ),
):
    """
    Dev/test endpoint to fan out an SSE message to subscribers.

    Protect with env var:
      - Set `EVENTS_PUBLISH_TOKEN=<secret>` in your environment.
      - Then call: POST /events/publish?token=<secret>

    Body example:
    {
      "channel": "public",
      "event": "meeting.started",
      "data": { "title": "Interview started", "join_url": "https://..." }
    }
    """
    if broker is None:
        raise HTTPException(status_code=501, detail="SSE broker not available")

    # Optional simple guard for non-prod usage
    required = os.getenv("EVENTS_PUBLISH_TOKEN", "").strip()
    if required and token != required:
      raise HTTPException(status_code=403, detail="Forbidden")

    delivered = await broker.publish(
        channel=body.channel.strip(),
        data=body.data,
        event=body.event,
        id=body.id,
        retry=body.retry,
    )
    return JSONResponse({"ok": True, "delivered": delivered})


@router.get("/health")
async def events_health():
    """
    Tiny health endpoint so you can quickly confirm the router is mounted.
    """
    if broker is None:
        # Router is up, broker is not.
        return JSONResponse({"ok": False, "broker": False})
    return JSONResponse({"ok": True, "broker": True})
