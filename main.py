# ✅ File: main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, Response, JSONResponse

from app.chatbot_router import router as chatbot_router
from app.auto_test_router import router as auto_test_router
from app.routers.upload_router import router as upload_router
from app.routers import auth_router
from app.routers import history_router
from app.routers import candidate_router
from app.utils.mongo import verify_mongo_connection  # ✅ DB check

# NEW: tests router (email invites + test lifecycle)
from app.routers.tests_router import router as tests_router

# ✅ NEW: interviews router (schedule interview + listings)
from app.routers.interviews_router import router as interviews_router

# ✅ NEW: tests history (attempts list + PDF reports)
from app.routers.tests_history_router import router as tests_history_router

# Optional: proctoring endpoints (start/heartbeat/snapshot/end)
try:
    from app.routers.proctor_router import router as proctor_router
except Exception:
    proctor_router = None  # graceful if file not present

# Optional: code runner endpoints (languages, dry-run, etc.)
try:
    from app.routers.runner_router import router as runner_router
except Exception:
    runner_router = None  # graceful if file not present

import uvicorn
import os
import base64
import json
import secrets
import inspect
from pathlib import Path
from typing import Optional, List, Any
from pydantic import BaseModel


app = FastAPI(
    title="SmartHirex API",
    description="AI-powered CV filtering, ML classification, and test automation.",
    version="2.0.0"
)

# ✅ CORS for frontend (dynamic but backward-compatible)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add FRONTEND_BASE_URL (if set)
_frontend_env = os.getenv("FRONTEND_BASE_URL")
if _frontend_env and _frontend_env not in origins:
    origins.append(_frontend_env)  # fixed: avoid undefined name and keep behavior

# Add extra origins from env (comma-separated), optional
_extra = os.getenv("CORS_EXTRA_ORIGINS", "")
if _extra:
    for o in [x.strip() for x in _extra.split(",") if x.strip()]:
        if o not in origins:
            origins.append(o)

_allow_all = os.getenv("CORS_ALLOW_ALL", "0").strip().lower() in {"1", "true", "yes", "on"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register API routers (existing)
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(auto_test_router, prefix="/auto-test")
app.include_router(upload_router, prefix="/upload")
app.include_router(auth_router.router, prefix="/auth")
app.include_router(history_router.router, prefix="/history")
app.include_router(candidate_router.router, prefix="/candidate")

# ✅ Register NEW tests router
app.include_router(tests_router, prefix="/tests")

# ✅ Register NEW interviews router (has internal prefix '/interviews')
app.include_router(interviews_router)

# ✅ Register NEW tests history router (internal prefix '/tests/history')
app.include_router(tests_history_router)

# ✅ (Optional) Proctor router
# NOTE: The router itself has NO internal prefix now; we add it here only once.
if proctor_router:
    app.include_router(proctor_router, prefix="/proctor")

# ✅ (Optional) Code runner router
# Exposed only if the module exists (and you can toggle behavior via env in the runner itself)
if runner_router:
    app.include_router(runner_router, prefix="/runner")

# ---------------------------
# Static mounts (non-breaking)
# ---------------------------
_root = Path(__file__).resolve().parent
_public_dir = Path(os.getenv("PUBLIC_DIR", _root / "public"))
_static_dir = Path(os.getenv("STATIC_DIR", _root / "static"))
_assets_dir = Path(os.getenv("ASSETS_DIR", _root / "assets"))

if _public_dir.is_dir():
    app.mount("/public", StaticFiles(directory=str(_public_dir)), name="public")
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

# ---------------------------------------------------------
# Default avatar fallback to eliminate repeated 404s
# ---------------------------------------------------------
_DEF_AVATAR_CANDIDATES = [
    _public_dir / "default-avatar.png",
    _static_dir / "default-avatar.png",
    _assets_dir / "default-avatar.png",
    _root / "default-avatar.png",
]

# 80x80 simple placeholder PNG (base64) — valid and minimal
_DEFAULT_AVATAR_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
)

@app.get("/default-avatar.png")
async def default_avatar() -> Response:
    for p in _DEF_AVATAR_CANDIDATES:
        if p.is_file():
            return FileResponse(str(p), media_type="image/png")
    return Response(content=base64.b64decode(_DEFAULT_AVATAR_B64), media_type="image/png")


# ---------------------------
# Small health endpoint
# ---------------------------
@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok", "service": "smarthirex-api"})


# ===========================
# Helpers (DB + tokens)
# ===========================
async def _get_db_optional() -> Optional[Any]:
    """
    Try to obtain a DB handle from app.utils.mongo (async or sync),
    but never crash if the helper isn't available.
    """
    try:
        from app.utils.mongo import get_db  # type: ignore
        maybe = get_db()
        return await maybe if inspect.isawaitable(maybe) else maybe
    except Exception:
        try:
            from app.utils.mongo import get_database  # type: ignore
            maybe = get_database()
            return await maybe if inspect.isawaitable(maybe) else maybe
        except Exception:
            return None


def _make_candidate_token(sub: str, email: Optional[str] = None) -> str:
    """
    Lightweight, app-local token (NOT a full JWT).
    Encodes role=candidate + subject + issued-at + random nonce.
    Frontend treats it as an opaque bearer token.
    """
    payload = {
        "role": "candidate",
        "sub": sub or "unknown",
        "email": email or None,
        "iat": int(os.environ.get("FAKE_JWT_IAT", "0")) or __import__("time").time(),
        "nonce": secrets.token_urlsafe(6),
    }
    b64 = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    return f"cand.{b64}.{secrets.token_urlsafe(8)}"


def _extract_bearer_or_cookie_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    # common cookie names used by frontend
    for key in ("access_token", "AUTH_TOKEN", "token", "authToken"):
        v = request.cookies.get(key)
        if v:
            return v
    return None


def _cookie_secure_default() -> bool:
    # secure cookies off for localhost by default; enable with COOKIE_SECURE=1
    return os.getenv("COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes", "on"}


# ===========================
# Req. 1 & 2 & 7 endpoints
# ===========================

class CandidateSessionRequest(BaseModel):
    # Accept either "test_token" or "token" (both optional names for the same thing)
    test_token: Optional[str] = None
    token: Optional[str] = None
    candidate_id: Optional[str] = None
    email: Optional[str] = None


@app.post("/auth/candidate-session")
async def create_candidate_session(body: CandidateSessionRequest, request: Request):
    """
    Create a candidate-scoped session from a test token, without creating a normal website user.
    Returns an opaque bearer token and also sets common cookies used by the frontend.
    """
    supplied = body.test_token or body.token
    if not supplied:
        return JSONResponse({"detail": "test_token required"}, status_code=400)

    subject = body.candidate_id or (body.email or "candidate")
    tok = _make_candidate_token(subject, body.email)

    # Optionally persist session (best effort)
    db = await _get_db_optional()
    if db:
        try:
            doc = {
                "kind": "candidate_session",
                "token": tok,
                "test_token": supplied,
                "candidate_id": body.candidate_id,
                "email": body.email,
                "created_at": __import__("datetime").datetime.utcnow(),
                "user_agent": request.headers.get("user-agent"),
                "ip": request.client.host if request.client else None,
            }
            ins = db["sessions"].insert_one(doc)
            if inspect.isawaitable(ins):
                await ins
        except Exception:
            # non-fatal
            pass

    # Set cookies commonly read by the frontend
    secure = _cookie_secure_default()
    resp = JSONResponse({"ok": True, "role": "candidate", "token": tok})
    resp.set_cookie("access_token", tok, path="/", httponly=False, samesite="lax", secure=secure)
    resp.set_cookie("AUTH_TOKEN", tok, path="/", httponly=False, samesite="lax", secure=secure)
    resp.set_cookie("role", "candidate", path="/", httponly=False, samesite="lax", secure=secure)
    return resp


@app.post("/logout")
async def logout(request: Request):
    """
    Invalidate server session (if stored) and expire auth cookies.
    """
    tok = _extract_bearer_or_cookie_token(request)

    # Best-effort server invalidation (if sessions collection exists)
    db = await _get_db_optional()
    if db and tok:
        try:
            res = db["sessions"].delete_many({"token": tok})
            if inspect.isawaitable(res):
                await res
        except Exception:
            # ignore failures; client cookies will still be cleared
            pass

    resp = JSONResponse({"ok": True})
    # Expire common auth cookies
    for c in ("access_token", "refresh_token", "AUTH_TOKEN", "Authorization", "session", "sessionid", "jwt", "authToken", "token", "role"):
        try:
            resp.delete_cookie(c, path="/")
        except Exception:
            pass
    return resp


class ResultsSaveRequest(BaseModel):
    ids: List[str]
    metadata: Optional[dict] = None
    source: Optional[str] = None


@app.post("/results/save")
async def save_filtered_results(body: ResultsSaveRequest, request: Request):
    """
    Persist selected results for the current actor (website user or candidate).
    Associates by token (opaque) and stores basic metadata.
    """
    if not body.ids:
        return JSONResponse({"detail": "ids array is required"}, status_code=400)

    owner_token = _extract_bearer_or_cookie_token(request)

    saved_id = None
    db = await _get_db_optional()
    if db:
        try:
            doc = {
                "kind": "saved_results",
                "ids": body.ids,
                "metadata": body.metadata or {},
                "source": body.source or "unknown",
                "owner_token": owner_token,
                "created_at": __import__("datetime").datetime.utcnow(),
                "user_agent": request.headers.get("user-agent"),
                "ip": request.client.host if request.client else None,
            }
            ins = db["saved_results"].insert_one(doc)
            if inspect.isawaitable(ins):
                ins = await ins
            saved_id = getattr(ins, "inserted_id", None)
        except Exception:
            # fall through to OK response without DB id
            pass

    return JSONResponse({"ok": True, "savedId": str(saved_id) if saved_id else None})


# ✅ Confirm MongoDB connection at startup
@app.on_event("startup")
async def startup_db_check():
    await verify_mongo_connection()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
