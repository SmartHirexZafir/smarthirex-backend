# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chatbot_router import router as chatbot_router
from app.auto_test_router import router as auto_test_router
from app.routers.upload_router import router as upload_router
from app.routers import auth_router

import uvicorn
import os

app = FastAPI(
    title="Chatbot Resume Filter + Auto Test System",
    description="API for chatbot-driven CV filtering, ML Resume Classifier, and AI test generation",
    version="2.0.0"
)

# ✅ CORS CONFIG for Frontend <-> Backend connection
origins = [
    "http://localhost:3000",  # Adjust this to your frontend URL or domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Which frontend origins are allowed
    allow_credentials=True,
    allow_methods=["*"],               # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],               # Allow all headers (esp. for auth tokens)
)

# ✅ Register routers
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(auto_test_router, prefix="/auto-test")
app.include_router(upload_router, prefix="/upload")
app.include_router(auth_router.router, prefix="/auth")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
