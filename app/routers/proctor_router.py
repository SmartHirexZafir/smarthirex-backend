# app/routers/proctor_router.py
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import os
import uuid
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
from fastapi import Body, UploadFile, File, Form, Depends, Query, Request

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import jwt

# Use the same DB handle style as the rest of your app (tests_router, etc.)
from app.utils.mongo import db  # AsyncIOMotorDatabase
from app.routers.auth_router import get_current_user
from app.utils.datetime_serialization import serialize_utc, serialize_utc_any

# IMPORTANT:
# Do NOT set a prefix here, because main.py already includes this router with prefix="/proctor".
# Having a prefix in both places causes routes like "/proctor/proctor/start" -> 404.
router = APIRouter(tags=["proctoring"])
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET must be set in environment")
JWT_ALG = "HS256"
PROCTOR_MEDIA_TOKEN_TTL_SEC = int(os.getenv("PROCTOR_MEDIA_TOKEN_TTL_SEC", "900") or "900")
MAX_SNAPSHOT_BYTES = int(os.getenv("PROCTOR_MAX_SNAPSHOT_BYTES", str(2 * 1024 * 1024)) or str(2 * 1024 * 1024))
MAX_VIDEO_BYTES = int(os.getenv("PROCTOR_MAX_VIDEO_BYTES", str(300 * 1024 * 1024)) or str(300 * 1024 * 1024))
VIDEOS_DIR = Path(os.getenv("PROCTOR_VIDEOS_DIR", "videos")).resolve()


# ---------- Pydantic models ----------

class StartProctorRequest(BaseModel):
    test_id: str = Field(..., description="ID of the test")
    candidate_id: str = Field(..., description="Candidate ID")
    # Optional token (frontend may pass it; we just store for traceability)
    token: Optional[str] = Field(None, description="Candidate test token")


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
    candidate_token: Optional[str] = None


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
    candidate_token: Optional[str] = None


class EndProctorRequest(BaseModel):
    session_id: str = Field(..., description="Proctoring session ID")
    reason: Optional[str] = None
    candidate_token: Optional[str] = None


# ---------- Helpers ----------

def _now_iso() -> str:
    return serialize_utc(dt.datetime.now(dt.timezone.utc))


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _normalize_iso_utc(value: Any) -> Optional[str]:
    return serialize_utc_any(value)


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


def _media_token(resource: str, resource_id: str, recruiter_id: str, ttl_sec: int = PROCTOR_MEDIA_TOKEN_TTL_SEC) -> str:
    now = _utc_now()
    payload = {
        "scope": "proctor_media",
        "resource": resource,
        "resource_id": resource_id,
        "recruiter_id": recruiter_id,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(seconds=ttl_sec)).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _verify_media_token(access_token: str, resource: str, resource_id: str, recruiter_id: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(access_token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid or expired access token")
    if payload.get("scope") != "proctor_media":
        raise HTTPException(status_code=403, detail="Invalid media access scope")
    if payload.get("resource") != resource or payload.get("resource_id") != resource_id:
        raise HTTPException(status_code=403, detail="Access token does not match requested resource")
    if str(payload.get("recruiter_id") or "") != str(recruiter_id or ""):
        raise HTTPException(status_code=403, detail="Access token not valid for this recruiter")
    return payload


async def _assert_recruiter_owns_candidate(candidate_id: str, current_user: Any) -> None:
    owner_id = str(getattr(current_user, "id", None) or "")
    if not owner_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    cand = await db.parsed_resumes.find_one({"_id": candidate_id, "ownerUserId": owner_id}, {"_id": 1})
    if not cand:
        raise HTTPException(status_code=403, detail="Not authorized for this candidate")


async def _validate_candidate_test_token(candidate_token: str, test_id: str, candidate_id: str) -> None:
    tok = (candidate_token or "").strip()
    if not tok:
        raise HTTPException(status_code=401, detail="Missing candidate token")
    test = await db.tests.find_one({"token": tok}, {"_id": 1, "candidateId": 1, "status": 1})
    if not test:
        raise HTTPException(status_code=401, detail="Invalid candidate token")
    if str(test.get("_id")) != str(test_id) or str(test.get("candidateId")) != str(candidate_id):
        raise HTTPException(status_code=403, detail="Candidate token does not match this session")


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
    if not req.token:
        raise HTTPException(status_code=401, detail="Candidate token required")

    # Validate token corresponds to this candidate/test pair.
    await _validate_candidate_test_token(req.token, req.test_id, req.candidate_id)
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
        "meta": {"candidate_token_sha256": hashlib.sha256(req.token.encode("utf-8")).hexdigest()},
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
    candidate_token = None
    if isinstance(payload, dict):
        session_id = (
            payload.get("session_id")
            or (payload.get("extra") or {}).get("session_id")
            or payload.get("testSessionId")
        )
        candidate_token = payload.get("candidate_token") or (payload.get("extra") or {}).get("candidate_token")
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

    session = await _get_session_by_any_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await _validate_candidate_test_token(
        str(candidate_token or ""),
        str(session.get("test_id") or ""),
        str(session.get("candidate_id") or ""),
    )
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

    await _validate_candidate_test_token(
        str(req.candidate_token or ""),
        str(session.get("test_id") or ""),
        str(session.get("candidate_id") or ""),
    )
    payload_b64 = _strip_data_url_prefix(req.image_base64)
    # Validate base64
    try:
        decoded = base64.b64decode(payload_b64, validate=True)
        if len(decoded) > MAX_SNAPSHOT_BYTES:
            raise HTTPException(status_code=413, detail="Snapshot exceeds max allowed size")
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
    session = await _get_session_by_any_id(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await _validate_candidate_test_token(
        str(req.candidate_token or ""),
        str(session.get("test_id") or ""),
        str(session.get("candidate_id") or ""),
    )
    now = _now_iso()
    update = {
        "$set": {
            "ended_at": now,
            "active": False,
            "end_reason": req.reason or "ended_by_client",
        }
    }

    matched = await _update_session_by_any_id(req.session_id, update)

    return {"status": "ended", "time": now}


# ✅ NEW: Get monitoring data for a candidate
@router.get("/candidate/{candidate_id}")
async def get_candidate_monitoring(candidate_id: str, current=Depends(get_current_user)):
    """
    Get all monitoring data (sessions, snapshots, behavior) for a specific candidate.
    Used by the Test Screening tab in Candidate Profile.
    """
    await _assert_recruiter_owns_candidate(candidate_id, current)
    recruiter_id = str(getattr(current, "id", None) or "")

    # Get all proctoring sessions for this candidate
    sessions = []
    async for session in db.proctor_sessions.find({"candidate_id": candidate_id}).sort("started_at", -1):
        session_id = str(session.get("_id", ""))
        
        # Get snapshots metadata for this session (exclude heavy inline payloads)
        snapshots = []
        async for snap in db.proctor_snapshots.find({"session_id": session_id}).sort("captured_at", -1):
            snap_id = str(snap.get("_id", ""))
            access_token = _media_token("snapshot", snap_id, recruiter_id)
            snapshots.append({
                "id": snap_id,
                "captured_at": _normalize_iso_utc(snap.get("captured_at")) or snap.get("captured_at"),
                "width": snap.get("width"),
                "height": snap.get("height"),
                "snapshot_url": f"/proctor/snapshot/content/{snap_id}?access_token={access_token}",
            })
        
        # Detect suspicious behavior
        suspicious_actions = []
        if session.get("last_visibility") is False:
            suspicious_actions.append("Tab hidden/backgrounded")
        if session.get("last_focus") is False:
            suspicious_actions.append("Window lost focus")
        if session.get("last_camera_status") in {"idle", "degraded", "error"}:
            suspicious_actions.append(f"Camera issue: {session.get('last_camera_status')}")
        if session.get("last_page_url") and "test" not in (session.get("last_page_url") or "").lower():
            suspicious_actions.append("Navigated away from test page")
        
        video_id = session.get("video_id")
        video_access = _media_token("video", str(video_id), recruiter_id) if video_id else None
        sessions.append({
            "session_id": session_id,
            "test_id": session.get("test_id"),
            "candidate_id": session.get("candidate_id"),
            "started_at": _normalize_iso_utc(session.get("started_at")) or session.get("started_at"),
            "ended_at": _normalize_iso_utc(session.get("ended_at")) or session.get("ended_at"),
            "active": session.get("active", False),
            "last_heartbeat_at": _normalize_iso_utc(session.get("last_heartbeat_at")) or session.get("last_heartbeat_at"),
            "last_visibility": session.get("last_visibility"),
            "last_focus": session.get("last_focus"),
            "last_camera_status": session.get("last_camera_status"),
            "last_camera_meta": session.get("last_camera_meta"),
            "last_page_url": session.get("last_page_url"),
            "snapshots": snapshots,
            "suspicious_actions": suspicious_actions,
            "suspicious_count": len(suspicious_actions),
            # Include video info if available
            "video_id": session.get("video_id"),
            "video_uploaded_at": _normalize_iso_utc(session.get("video_uploaded_at")) or session.get("video_uploaded_at"),
            "video_info_url": f"/proctor/video/info/{video_id}?access_token={video_access}" if video_id and video_access else None,
            "video_download_url": f"/proctor/video/download/{video_id}?access_token={video_access}" if video_id and video_access else None,
        })
    
    return {
        "candidate_id": candidate_id,
        "sessions": sessions,
        "total_sessions": len(sessions),
        "total_snapshots": sum(len(s.get("snapshots", [])) for s in sessions),
    }


# ✅ NEW: Video upload endpoint
@router.post("/video/upload")
async def upload_video(
    video: UploadFile = File(...),
    session_id: str = Form(...),
    test_id: str = Form(...),
    candidate_id: str = Form(...),
    candidate_token: str = Form(...),
    uploaded_at: Optional[str] = Form(None),
):
    """
    Upload a video blob from frontend and store it securely.
    
    Requirements:
    - Validate session exists and is for this test/candidate
    - Store video file with secure naming
    - Create MongoDB record linking video to session
    - Return download token for recruiter access
    """
    # Validate session exists
    session = await _get_session_by_any_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Proctoring session not found")

    # Cross-check test_id and candidate_id match session
    if session.get("test_id") != test_id or session.get("candidate_id") != candidate_id:
        raise HTTPException(
            status_code=403,
            detail="Video does not match the current session"
        )

    await _validate_candidate_test_token(candidate_token, test_id, candidate_id)

    try:
        # Stream to disk in chunks to avoid loading full file into memory.
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_id = uuid.uuid4().hex
        video_filename = f"{video_id}.webm"
        video_path = VIDEOS_DIR / video_filename

        file_size = 0
        with open(video_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_VIDEO_BYTES:
                    f.close()
                    try:
                        video_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail="Video exceeds max allowed size")
                f.write(chunk)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Video file is empty")

        # Create MongoDB record
        video_doc = {
            "_id": video_id,
            "session_id": str(session.get("_id", session_id)),
            "test_id": test_id,
            "candidate_id": candidate_id,
            "filename": video_filename,
            "file_path": str(video_path),
            "file_size": file_size,
            "mime_type": video.content_type or "video/webm",
            "uploaded_at": uploaded_at or _now_iso(),
            "created_at": _now_iso(),
        }

        await db.proctor_videos.insert_one(video_doc)

        # Update session with video reference
        await _update_session_by_any_id(session_id, {
            "$set": {
                "video_id": video_id,
                "video_uploaded_at": _now_iso(),
            }
        })

        return {
            "status": "ok",
            "video_id": video_id,
            "file_size": file_size,
            "message": "Video uploaded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload video: {str(e)[:100]}"
        )


async def _finalize_video_upload_record(
    upload_doc: Dict[str, Any],
    *,
    uploaded_at: Optional[str] = None,
) -> Dict[str, Any]:
    video_id = uuid.uuid4().hex
    video_filename = f"{video_id}.webm"
    src_path = Path(str(upload_doc.get("file_path") or "")).resolve()
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Upload file not found")
    final_path = VIDEOS_DIR / video_filename
    try:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        os.replace(str(src_path), str(final_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to finalize upload file: {str(e)[:100]}")

    file_size = int(upload_doc.get("bytes_received") or 0)
    video_doc = {
        "_id": video_id,
        "session_id": upload_doc.get("session_id"),
        "test_id": upload_doc.get("test_id"),
        "candidate_id": upload_doc.get("candidate_id"),
        "filename": video_filename,
        "file_path": str(final_path),
        "file_size": file_size,
        "mime_type": upload_doc.get("mime_type") or "video/webm",
        "uploaded_at": uploaded_at or _now_iso(),
        "created_at": _now_iso(),
    }
    await db.proctor_videos.insert_one(video_doc)
    await _update_session_by_any_id(
        str(upload_doc.get("session_id") or ""),
        {"$set": {"video_id": video_id, "video_uploaded_at": _now_iso()}},
    )
    await db.proctor_video_uploads.update_one(
        {"_id": upload_doc["_id"]},
        {"$set": {"status": "finalized", "final_video_id": video_id, "finalized_at": _now_iso()}},
    )
    return {"video_id": video_id, "file_size": file_size}


@router.post("/video/upload/chunk")
async def upload_video_chunk(
    chunk: UploadFile = File(...),
    session_id: str = Form(...),
    test_id: str = Form(...),
    candidate_id: str = Form(...),
    candidate_token: str = Form(...),
    upload_id: Optional[str] = Form(None),
):
    session = await _get_session_by_any_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Proctoring session not found")
    if session.get("test_id") != test_id or session.get("candidate_id") != candidate_id:
        raise HTTPException(status_code=403, detail="Chunk does not match the current session")
    await _validate_candidate_test_token(candidate_token, test_id, candidate_id)

    up_id = str(upload_id or uuid.uuid4().hex)
    upload_doc = await db.proctor_video_uploads.find_one({"_id": up_id})
    if not upload_doc:
        tmp_path = (VIDEOS_DIR / f"upload_{up_id}.part").resolve()
        await db.proctor_video_uploads.insert_one(
            {
                "_id": up_id,
                "session_id": str(session.get("_id") or session_id),
                "test_id": test_id,
                "candidate_id": candidate_id,
                "file_path": str(tmp_path),
                "bytes_received": 0,
                "status": "active",
                "mime_type": chunk.content_type or "video/webm",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        )
        upload_doc = await db.proctor_video_uploads.find_one({"_id": up_id})
    if not upload_doc:
        raise HTTPException(status_code=500, detail="Failed to initialize upload session")
    if upload_doc.get("status") != "active":
        raise HTTPException(status_code=409, detail="Upload session is not active")

    file_path = Path(str(upload_doc.get("file_path") or "")).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_received = int(upload_doc.get("bytes_received") or 0)
    chunk_size = 0
    try:
        with open(file_path, "ab") as f:
            while True:
                buf = await chunk.read(1024 * 1024)
                if not buf:
                    break
                chunk_size += len(buf)
                bytes_received += len(buf)
                if bytes_received > MAX_VIDEO_BYTES:
                    await db.proctor_video_uploads.update_one(
                        {"_id": up_id},
                        {"$set": {"status": "failed", "updated_at": _now_iso(), "error": "max_size_exceeded"}},
                    )
                    try:
                        file_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail="Video exceeds max allowed size")
                f.write(buf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk write failed: {str(e)[:100]}")

    await db.proctor_video_uploads.update_one(
        {"_id": up_id},
        {"$set": {"bytes_received": bytes_received, "updated_at": _now_iso()}},
    )
    return {"status": "ok", "upload_id": up_id, "chunk_size": chunk_size, "bytes_received": bytes_received}


@router.post("/video/upload/finalize")
async def finalize_video_upload(
    upload_id: str = Form(...),
    session_id: str = Form(...),
    test_id: str = Form(...),
    candidate_id: str = Form(...),
    candidate_token: str = Form(...),
    uploaded_at: Optional[str] = Form(None),
):
    session = await _get_session_by_any_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Proctoring session not found")
    if session.get("test_id") != test_id or session.get("candidate_id") != candidate_id:
        raise HTTPException(status_code=403, detail="Finalize does not match the current session")
    await _validate_candidate_test_token(candidate_token, test_id, candidate_id)
    upload_doc = await db.proctor_video_uploads.find_one({"_id": upload_id})
    if not upload_doc:
        raise HTTPException(status_code=404, detail="Upload session not found")
    if str(upload_doc.get("session_id")) != str(session.get("_id")):
        raise HTTPException(status_code=403, detail="Upload session mismatch")
    if int(upload_doc.get("bytes_received") or 0) <= 0:
        raise HTTPException(status_code=400, detail="No uploaded video data to finalize")
    if upload_doc.get("status") == "finalized" and upload_doc.get("final_video_id"):
        return {"status": "ok", "video_id": upload_doc.get("final_video_id"), "file_size": int(upload_doc.get("bytes_received") or 0)}
    out = await _finalize_video_upload_record(upload_doc, uploaded_at=uploaded_at)
    return {"status": "ok", **out}


# ✅ NEW: Secure video download endpoint (recruiter only)
@router.get("/video/download/{video_id}")
async def download_video(video_id: str, access_token: str = Query(...), current=Depends(get_current_user)):
    """
    Download a recorded test video for recruiter viewing.
    
    Security:
    - Validates token matches the video's access_token
    - Returns 404 if token invalid/missing
    - Streams video file directly
    """
    from fastapi.responses import FileResponse
    
    recruiter_id = str(getattr(current, "id", None) or "")
    _verify_media_token(access_token, "video", video_id, recruiter_id)
    # Find video record
    video_doc = await db.proctor_videos.find_one({"_id": video_id})
    if not video_doc:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get file path
    file_path = video_doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    # Return file for download
    return FileResponse(
        file_path,
        media_type=video_doc.get("mime_type", "video/webm"),
        filename=f"test_{video_doc.get('test_id')}_recording.webm",
    )


# ✅ NEW: Get video metadata (for candidate profile)
@router.get("/video/info/{video_id}")
async def get_video_info(video_id: str, access_token: str = Query(...), current=Depends(get_current_user)):
    """
    Get video metadata (size, duration, status) for display in UI.
    Requires valid access token.
    """
    recruiter_id = str(getattr(current, "id", None) or "")
    _verify_media_token(access_token, "video", video_id, recruiter_id)
    video_doc = await db.proctor_videos.find_one({"_id": video_id})
    if not video_doc:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "video_id": video_id,
        "test_id": video_doc.get("test_id"),
        "candidate_id": video_doc.get("candidate_id"),
        "file_size": video_doc.get("file_size"),
        "mime_type": video_doc.get("mime_type"),
        "uploaded_at": _normalize_iso_utc(video_doc.get("uploaded_at")) or video_doc.get("uploaded_at"),
        "created_at": _normalize_iso_utc(video_doc.get("created_at")) or video_doc.get("created_at"),
    }


@router.get("/snapshot/content/{snapshot_id}")
async def get_snapshot_content(snapshot_id: str, access_token: str = Query(...), current=Depends(get_current_user)):
    recruiter_id = str(getattr(current, "id", None) or "")
    _verify_media_token(access_token, "snapshot", snapshot_id, recruiter_id)
    snap = await db.proctor_snapshots.find_one({"_id": snapshot_id})
    if not snap:
        try:
            from bson import ObjectId
            snap = await db.proctor_snapshots.find_one({"_id": ObjectId(snapshot_id)})
        except Exception:
            snap = None
    if not snap:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    payload_b64 = snap.get("image_base64") or ""
    try:
        raw = base64.b64decode(payload_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=500, detail="Snapshot data corrupted")
    from fastapi.responses import Response
    return Response(content=raw, media_type="image/jpeg")