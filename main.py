# ✅ File: main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
# If you haven't added the file, you can comment the two lines below.
try:
    from app.routers.proctor_router import router as proctor_router
except Exception:
    proctor_router = None  # graceful if file not present

import uvicorn
import os

app = FastAPI(
    title="SmartHirex API",
    description="AI-powered CV filtering, ML classification, and test automation.",
    version="2.0.0"
)

# ✅ CORS for frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Optionally add FRONTEND_BASE_URL from .env without breaking existing list
_frontend_env = os.getenv("FRONTEND_BASE_URL")
if _frontend_env and _frontend_env not in origins:
    origins.append(_frontend_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

# ✅ Confirm MongoDB connection at startup
@app.on_event("startup")
async def startup_db_check():
    await verify_mongo_connection()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
