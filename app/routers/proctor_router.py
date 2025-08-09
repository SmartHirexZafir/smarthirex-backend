# app/routers/proctor_router.py
from __future__ import annotations

import base64
import datetime as dt
from typing import Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from app.utils.mongo import get_db  # <-- adjust if your helper has a different name

router = APIRouter(prefix="/proctor", tags=["proctoring"])


# ---------- Pydantic models ----------

class StartProctorRequest(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    candidate_id: str = Field(..., description="Candidate ID")


class StartProctorResponse(BaseModel):
    session_id: str
    started_at: str


class HeartbeatRequest(BaseModel):
    session_id: str = Field(..., description="Proctoring session ID")
    # optional extra client telemetry
    page_visible: Optional[bool] = True
    focused: Optional[bool] = True


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


# ---------- Routes ----------

@router.post("/start", response_model=StartProctorResponse)
async def start_proctoring(req: StartProctorRequest):
    """
    Create a proctoring session for a given test & candidate.
    """
    db = await get_db()
    started_at = _now_iso()

    doc = {
        "test_id": req.test_id,
        "candidate_id": req.candidate_id,
        "started_at": started_at,
        "ended_at": None,
        "last_heartbeat_at": started_at,
        "active": True,
        "meta": {},
    }

    res = await db.proctor_sessions.insert_one(doc)
    return StartProctorResponse(session_id=str(res.inserted_id), started_at=started_at)


@router.post("/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    """
    Update the session's last-seen timestamp and (optionally) visibility/focus status.
    """
    db = await get_db()
    now = _now_iso()

    update = {
        "$set": {
            "last_heartbeat_at": now,
            "last_visibility": bool(req.page_visible),
            "last_focus": bool(req.focused),
        }
    }

    result = await db.proctor_sessions.update_one({"_id": req.session_id}, update)
    # If _id is an ObjectId in your DB, adapt this query to convert to ObjectId
    if result.matched_count == 0:
        # Try ObjectId fallback if needed
        try:
            from bson import ObjectId
            result = await db.proctor_sessions.update_one({"_id": ObjectId(req.session_id)}, update)
        except Exception:
            pass

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ok", "time": now}


@router.post("/snapshot")
async def snapshot(req: SnapshotRequest):
    """
    Receive a base64-encoded image and store it as part of the session.
    """
    db = await get_db()

    # Confirm session exists & active
    session = await db.proctor_sessions.find_one({"_id": req.session_id})
    if not session:
        # Try ObjectId fallback if needed
        try:
            from bson import ObjectId
            session = await db.proctor_sessions.find_one({"_id": ObjectId(req.session_id)})
        except Exception:
            session = None

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
    Close a session (mark inactive).
    """
    db = await get_db()
    now = _now_iso()

    update = {
        "$set": {
            "ended_at": now,
            "active": False,
            "end_reason": req.reason or "ended_by_client",
        }
    }

    result = await db.proctor_sessions.update_one({"_id": req.session_id}, update)
    # ObjectId fallback
    if result.matched_count == 0:
        try:
            from bson import ObjectId
            result = await db.proctor_sessions.update_one({"_id": ObjectId(req.session_id)}, update)
        except Exception:
            pass

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ended", "time": now}
