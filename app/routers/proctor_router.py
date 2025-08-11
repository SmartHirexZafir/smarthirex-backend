# app/routers/proctor_router.py
from __future__ import annotations

import base64
import datetime as dt
from typing import Optional, Any, Dict
from fastapi import Body

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

# Use the same DB handle style as the rest of your app (tests_router, etc.)
from app.utils.mongo import db  # AsyncIOMotorDatabase

# IMPORTANT:
# Do NOT set a prefix here, because main.py already includes this router with prefix="/proctor".
# Having a prefix in both places causes routes like "/proctor/proctor/start" -> 404.
router = APIRouter(tags=["proctoring"])


# ---------- Pydantic models ----------

class StartProctorRequest(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    candidate_id: str = Field(..., description="Candidate ID")
    # Optional token (frontend may pass it; we just store for traceability)
    token: Optional[str] = Field(None, description="Optional association token")


class StartProctorResponse(BaseModel):
    session_id: str
    started_at: str


class HeartbeatLegacy(BaseModel):
    """
    Legacy heartbeat shape (from original ProctorGuard.tsx):
    { session_id: str, page_visible?: bool, focused?: bool }
    """
    session_id: Optional[str] = Field(None, description="Proctoring session ID")
    page_visible: Optional[bool] = True
    focused: Optional[bool] = True


class HeartbeatBeacon(BaseModel):
    """
    New heartbeat shape (from ProctorHeartbeat.ts):
    {
      testSessionId?: string,
      userId?: string,
      ts?: string,
      pageUrl?: string,
      status?: "ok"|"degraded"|"idle"|"error",
      camera?: object,
      extra?: { session_id?: string, ... }
    }
    """
    testSessionId: Optional[str] = None
    userId: Optional[str] = None
    ts: Optional[str] = None
    pageUrl: Optional[str] = None
    status: Optional[str] = None
    camera: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

    @validator("status")
    def _status_ok(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.lower()
        if v2 not in {"ok", "degraded", "idle", "error"}:
            raise ValueError("status must be one of: ok, degraded, idle, error")
        return v2


class SnapshotRequest(BaseModel):
    session_id: str = Field(..., description="Proctoring session ID")
    # dataURL or raw base64 (we handle both)
    image_base64: str = Field(..., description="Base64 of PNG/JPEG frame (no headers OK)")
    # optional client info
    taken_at: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class EndProctorRequest(BaseModel):
    session_id: str = Field(..., description="Proctoring session ID")
    reason: Optional[str] = None


# ---------- Helpers ----------

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def _strip_data_url_prefix(b64: str) -> str:
    """
    Accepts either 'data:image/png;base64,AAA...' or plain 'AAA...'
    Returns only the raw base64 payload.
    """
    if "," in b64 and b64.strip().lower().startswith("data:"):
        return b64.split(",", 1)[1]
    return b64


async def _get_session_by_any_id(session_id: str):
    """
    Try to fetch a session by either plain string _id or ObjectId.
    Returns the session doc or None.
    """
    session = await db.proctor_sessions.find_one({"_id": session_id})
    if session:
        return session
    try:
        from bson import ObjectId
        oid = ObjectId(session_id)
        session = await db.proctor_sessions.find_one({"_id": oid})
        return session
    except Exception:
        return None


async def _update_session_by_any_id(session_id: str, update: Dict[str, Any]):
    """
    Update by plain string id, fallback to ObjectId.
    Returns matched_count.
    """
    result = await db.proctor_sessions.update_one({"_id": session_id}, update)
    if result.matched_count > 0:
        return result.matched_count
    try:
        from bson import ObjectId
        result = await db.proctor_sessions.update_one({"_id": ObjectId(session_id)}, update)
        return result.matched_count
    except Exception:
        return 0


# ---------- Routes ----------

@router.post("/start", response_model=StartProctorResponse)
async def start_proctoring(req: StartProctorRequest):
    """
    Create a proctoring session for a given test & candidate.
    """
    started_at = _now_iso()

    doc = {
        "test_id": req.test_id,
        "candidate_id": req.candidate_id,
        "started_at": started_at,
        "ended_at": None,
        "last_heartbeat_at": started_at,
        "active": True,
        # Last-known client telemetry (populated by heartbeats)
        "last_visibility": None,
        "last_focus": None,
        "last_page_url": None,
        "last_camera_status": None,     # "ok" | "degraded" | "idle" | "error"
        "last_camera_meta": None,       # arbitrary dict from client
        "last_client_ts": None,         # client-reported iso ts (if provided)
        # Extra metadata
        "meta": {"token": req.token} if req.token else {},
    }

    res = await db.proctor_sessions.insert_one(doc)
    return StartProctorResponse(session_id=str(res.inserted_id), started_at=started_at)



@router.post("/heartbeat")
async def heartbeat(payload: dict = Body(...)):
    """
    Accepts either legacy {session_id, page_visible?, focused?}
    OR new beacon {testSessionId?, userId?, ts?, pageUrl?, status?, camera?, extra:{session_id?}}
    """
    now = _now_iso()

    # resolve session_id from multiple shapes
    session_id = None
    if isinstance(payload, dict):
        session_id = (
            payload.get("session_id")
            or (payload.get("extra") or {}).get("session_id")
            or payload.get("testSessionId")
        )
    session_id = (str(session_id or "").strip()) or None
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    # build update set
    set_fields = {"last_heartbeat_at": now}

    # legacy flags
    if "page_visible" in payload:
        set_fields["last_visibility"] = bool(payload.get("page_visible"))
    if "focused" in payload:
        set_fields["last_focus"] = bool(payload.get("focused"))

    # beacon fields
    if "pageUrl" in payload:
        set_fields["last_page_url"] = str(payload.get("pageUrl") or "")
    if "status" in payload:
        set_fields["last_camera_status"] = str(payload.get("status") or "")
    if "camera" in payload:
        cam = payload.get("camera")
        # keep as-is if JSON-serializable
        if isinstance(cam, dict):
            set_fields["last_camera_meta"] = cam
    if "ts" in payload:
        set_fields["last_client_ts"] = str(payload.get("ts") or "")

    matched = await _update_session_by_any_id(session_id, {"$set": set_fields})
    if matched == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ok", "time": now}



@router.post("/snapshot")
async def snapshot(req: SnapshotRequest):
    """
    Receive a base64-encoded image and store it as part of the session.
    Accepts either the stringified ObjectId or the raw ObjectId as session_id.
    """
    # Confirm session exists & active
    session = await _get_session_by_any_id(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("active") is not True:
        raise HTTPException(status_code=400, detail="Session is not active")

    payload_b64 = _strip_data_url_prefix(req.image_base64)
    # Validate base64
    try:
        _ = base64.b64decode(payload_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    snap_doc = {
        "session_id": str(session["_id"]),
        "test_id": session.get("test_id"),
        "candidate_id": session.get("candidate_id"),
        "captured_at": req.taken_at or _now_iso(),
        "image_base64": payload_b64,  # keeping simple; can move to GridFS later
        "width": req.width,
        "height": req.height,
    }

    await db.proctor_snapshots.insert_one(snap_doc)

    # Touch heartbeat as well
    await db.proctor_sessions.update_one(
        {"_id": session["_id"]},
        {"$set": {"last_heartbeat_at": _now_iso()}}
    )

    return {"status": "stored"}


@router.post("/end")
async def end_proctoring(req: EndProctorRequest):
    """
    Close a session (mark inactive). Accepts either the stringified ObjectId or the raw ObjectId.
    """
    now = _now_iso()
    update = {
        "$set": {
            "ended_at": now,
            "active": False,
            "end_reason": req.reason or "ended_by_client",
        }
    }

    matched = await _update_session_by_any_id(req.session_id, update)
    if matched == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ended", "time": now}
