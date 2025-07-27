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

import uvicorn
import os

app = FastAPI(
    title="SmartHirex API",
    description="AI-powered CV filtering, ML classification, and test automation.",
    version="2.0.0"
)

# ✅ CORS for frontend
origins = [
    "http://localhost:3000",  # adjust if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register API routers
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(auto_test_router, prefix="/auto-test")
app.include_router(upload_router, prefix="/upload")
app.include_router(auth_router.router, prefix="/auth")
app.include_router(history_router.router, prefix="/history")
app.include_router(candidate_router.router, prefix="/candidate")

# ✅ Confirm MongoDB connection at startup
@app.on_event("startup")
async def startup_db_check():
    await verify_mongo_connection()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
