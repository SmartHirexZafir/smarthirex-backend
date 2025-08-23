# app/utils/sse_bus.py
"""
Lightweight in-process Server-Sent Events (SSE) broker for FastAPI.

Usage (router example)
----------------------
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.utils.sse_bus import broker, format_sse

router = APIRouter(prefix="/events")

@router.get("/subscribe/{channel}")
async def sse_subscribe(channel: str):
    # IMPORTANT: set correct media_type and disable response buffering/proxying in deployment
    async def event_generator():
        async for msg in broker.subscribe(channel):
            yield msg
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Somewhere else in your code when you want to notify clients:
await broker.publish(
    channel="user:123",
    event="meeting_starting",
    data={"minutes": 5, "meetingUrl": "https://..."}
)

Notes
-----
- This broker is **in-memory** (per process). If you run multiple workers (e.g., uvicorn --workers 4),
  each worker will have its own broker. For multi-worker distribution you’ll need a backplane
  (e.g., Redis pub/sub) – this file keeps things simple and local.
- Sends keepalive heartbeats (event: ping) every `heartbeat_interval` seconds so proxies
  don’t time out idle connections.
- Safe to use across endpoints in the same process.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Set


# ---------------------------
# SSE formatting helpers
# ---------------------------

def _to_json(obj: Any) -> str:
    """Serialize to JSON with a safe default."""
    try:
        return json.dumps(obj, separators=(",", ":"), default=_json_default)
    except Exception:
        # Best-effort string fallback
        return json.dumps(str(obj))


def _json_default(o: Any) -> str:
    """Fallback serializer for datetimes and other objects."""
    try:
        # Datetime-like objects
        import datetime as _dt  # local import to avoid hard dep at module import
        if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
            # ISO with Z for UTC when tzinfo is UTC
            if isinstance(o, _dt.datetime) and (o.tzinfo is not None and o.tzinfo.utcoffset(o) == _dt.timedelta(0)):
                return o.isoformat().replace("+00:00", "Z")
            return o.isoformat()
    except Exception:
        pass
    return str(o)


def _sse_escape_data(s: str) -> str:
    """
    SSE data can be multi-line; each line must be prefixed with "data: ".
    Also ensure we end with a double newline to flush the event.
    """
    # Normalize line endings and prefix lines with 'data: '
    lines = s.splitlines() or [s]
    return "".join(f"data: {line}\n" for line in lines)


def format_sse(*, data: Any, event: Optional[str] = None, id: Optional[str] = None, retry: Optional[int] = None) -> str:
    """
    Build an SSE string block.

    Example:
        format_sse(data={"x": 1}, event="update", id="abc", retry=10000)
    """
    parts: list[str] = []

    if event:
        parts.append(f"event: {event}\n")
    if id:
        parts.append(f"id: {id}\n")
    if retry is not None:
        parts.append(f"retry: {int(retry)}\n")

    payload = data if isinstance(data, str) else _to_json(data)
    parts.append(_sse_escape_data(payload))
    parts.append("\n")  # end of message (blank line)
    return "".join(parts)


# ---------------------------
# Broker
# ---------------------------

@dataclass(eq=False)
class _Subscriber:
    """A single client connection."""
    queue: "asyncio.Queue[str]" = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=lambda: time.time())
    last_seen: float = field(default_factory=lambda: time.time())

    def touch(self) -> None:
        self.last_seen = time.time()


class SSEBroker:
    """
    In-memory broker for SSE channels.

    - `subscribe(channel)` returns an async generator yielding SSE-formatted strings.
    - `publish(channel, ...)` enqueues an event to all current subscribers of that channel.
    """

    def __init__(self, *, heartbeat_interval: int = 15, drop_oldest_on_backpressure: bool = True) -> None:
        self._channels: Dict[str, Set[_Subscriber]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = int(max(5, heartbeat_interval))
        self._drop_oldest = drop_oldest_on_backpressure

    # ------------- subscription -------------

    async def subscribe(self, channel: str) -> AsyncIterator[str]:
        """
        Async generator for a single client subscription.

        Yields SSE-formatted strings until the client disconnects or generator is closed.
        """
        sub = _Subscriber()
        heartbeat_task: Optional[asyncio.Task] = None

        async with self._lock:
            self._channels.setdefault(channel, set()).add(sub)

        try:
            # Initial comment to open the stream quickly (optional)
            yield ": connected\n\n"

            # Start per-subscriber heartbeat
            heartbeat_task = asyncio.create_task(self._heartbeat_publisher(sub))

            while True:
                msg = await sub.queue.get()
                sub.touch()
                yield msg
        except asyncio.CancelledError:
            # client disconnected or server shut down
            raise
        finally:
            if heartbeat_task:
                heartbeat_task.cancel()
                with contextlib.suppress(Exception):
                    await heartbeat_task
            await self._remove_subscriber(channel, sub)

    async def _remove_subscriber(self, channel: str, sub: _Subscriber) -> None:
        async with self._lock:
            subs = self._channels.get(channel)
            if subs and sub in subs:
                subs.remove(sub)
                if not subs:
                    # Clean up empty channels
                    self._channels.pop(channel, None)

    # ------------- publishing -------------

    async def publish(
        self,
        *,
        channel: str,
        data: Any,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> int:
        """
        Publish an event to a channel. Returns number of subscribers notified.
        """
        message = format_sse(data=data, event=event, id=id, retry=retry)
        async with self._lock:
            subs = list(self._channels.get(channel, set()))
        delivered = 0
        for sub in subs:
            if await self._offer(sub.queue, message):
                delivered += 1
        return delivered

    async def broadcast(
        self,
        *,
        data: Any,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> int:
        """
        Publish an event to ALL channels. Returns number of subscribers notified.
        """
        message = format_sse(data=data, event=event, id=id, retry=retry)
        async with self._lock:
            # Flatten all subscribers across channels
            all_subs = {sub for subs in self._channels.values() for sub in subs}
        delivered = 0
        for sub in all_subs:
            if await self._offer(sub.queue, message):
                delivered += 1
        return delivered

    async def _offer(self, q: "asyncio.Queue[str]", item: str) -> bool:
        """
        Try enqueue without blocking. If full and configured to drop oldest,
        clear one item and retry once.
        """
        try:
            q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            if self._drop_oldest:
                try:
                    _ = q.get_nowait()  # drop one
                except Exception:
                    return False
                try:
                    q.put_nowait(item)
                    return True
                except Exception:
                    return False
            return False

    # ------------- heartbeat -------------

    async def _heartbeat_publisher(self, sub: _Subscriber) -> None:
        """
        Periodically push a small SSE 'ping' event so connections remain open through proxies.
        """
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    ping = format_sse(data="💓", event="ping")
                    sub.queue.put_nowait(ping)
                except asyncio.QueueFull:
                    # If the client isn't reading, drop heartbeat silently
                    pass
        except asyncio.CancelledError:
            # Normal tear-down
            return


# Singleton broker used app-wide
# Import this in your routers: `from app.utils.sse_bus import broker`
broker = SSEBroker()


# ------------- small stdlib-only contextlib shim -------------

# (Avoid importing contextlib at module top to keep import overhead minimal.)
import contextlib  # noqa: E402  (placed at end on purpose)
