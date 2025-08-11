# ✅ File: main.py

from fastapi import FastAPI
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
from pathlib import Path


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

# 80x80 simple placeholder PNG (base64)
_DEFAULT_AVATAR_B64 = (
    b"iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAQAAABuC6+eAAAACXBIWXMAAAsTAAALEwEAmpwYAAAB"
    b"VUlEQVR4nO3UsQ3CMBQF0W9a0o9a0o8F7l2Sg0pQF3b4mF3mWw7jVq2p6o6w2QXfFZqg8JmQm+eZ"
    b"eQ4n6q9H3f5vXh4M2Gf8kYw3k0v86nK8f6v2oO5w2mW3Sg5K0Bf8It6mCwqkQ8wF1mC8Q5n2Qf0C"
    b"0N9E1gqS3Q8oA3oH0vH8q3wX8gDtxr4V0G4yJc1Cz8qF3p9y7a3d7M0y3lJj8l1Fh7k2Kq8cD8fQ"
    b"mQkQkQkQkQkQkQkQkQkQkQkQkQkQkQkQkQkQmQkQkQkQkQkQkQkQmQm3vN8k4g2O3J0c8c2l0"
    b"QKZs7f3jzq7r8CwJgq7z2K2kCk7w2b3kQ8OaU8c3cRyOuwcYJwCw7jO6b8E1g3wQkQkQkQkQkQkQ"
    b"kQkQkQkQkQkQkQkQkQkQkQkQkQkQmQkQkQkQkQkQkQkQmQn0P7M2v4Qe6mQAAAAAElFTkSuQmCC"
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

# ✅ Confirm MongoDB connection at startup
@app.on_event("startup")
async def startup_db_check():
    await verify_mongo_connection()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
