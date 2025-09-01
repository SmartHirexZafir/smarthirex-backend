# app/routers/auth_router.py

from fastapi import APIRouter, HTTPException, Request, status, Body, Query
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from app.utils.mongo import db
from app.utils.emailer import send_verification_email
from dotenv import load_dotenv
from types import SimpleNamespace
from typing import Optional
from datetime import datetime, timedelta, timezone  # ✅ include timezone for aware datetimes
import jwt, uuid, os

load_dotenv()

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET", "smarthirex-secret")
ALGORITHM = "HS256"
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
VERIFICATION_TTL_HOURS = 24
JWT_EXPIRES_HOURS = int(os.getenv("JWT_EXPIRES_HOURS", "168"))  # ✅ default 7 days

# ----------- Models -----------

class SignupRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    password: str
    company: str
    jobTitle: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ----------- Helpers -----------

def _now_utc() -> datetime:
    """Timezone-aware UTC now for consistent storage and comparisons."""
    return datetime.now(timezone.utc)

def _verify_link(token: str) -> str:
    """
    Link sent to the user's email. It points to the Next.js page:
    /verify/[token] which will call the backend and then redirect to login.
    """
    return f"{FRONTEND_BASE_URL}/verify/{token}"

async def _create_verify_token(user_id: str) -> str:
    token = str(uuid.uuid4())
    await db.email_verification_tokens.insert_one({
        "user_id": user_id,
        "token": token,
        "expires_at": _now_utc() + timedelta(hours=VERIFICATION_TTL_HOURS),  # ✅ aware
        "used": False,
        "created_at": _now_utc(),  # ✅ aware
    })
    return token

async def _verify_token_and_mark_used(token: str):
    tok = await db.email_verification_tokens.find_one({"token": token})
    if (not tok) or tok.get("used"):
        raise HTTPException(status_code=400, detail="Invalid or used token")
    if tok["expires_at"] < _now_utc():  # ✅ aware comparison
        raise HTTPException(status_code=400, detail="Token expired")

    # mark user as verified + token as used
    await db.users.update_one({"_id": tok["user_id"]}, {"$set": {"is_verified": True}})
    await db.email_verification_tokens.update_one({"_id": tok["_id"]}, {"$set": {"used": True}})

def _wants_json(request: Request, redirect_param: Optional[str] = None) -> bool:
    """
    If the client is a fetch/XHR (Accept includes application/json) OR
    explicitly asks with ?redirect=0, return JSON instead of Redirect.
    """
    if redirect_param is not None and redirect_param == "0":
        return True
    accept = (request.headers.get("accept") or "").lower()
    return "application/json" in accept

# ----------- Signup -----------

@router.post("/signup")
async def signup(data: SignupRequest):
    # ✅ normalize email to lowercase to avoid case-variant duplicates
    email_lc = str(data.email).strip().lower()

    existing = await db.users.find_one({"email": email_lc})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed = pwd_context.hash(data.password)
    user = data.dict()
    user["_id"] = str(uuid.uuid4())
    user["email"] = email_lc  # ✅ store lowercased
    user["password"] = hashed
    user["is_verified"] = False
    user["created_at"] = _now_utc()  # ✅ aware timestamp

    await db.users.insert_one(user)

    # create token + send verification email (no login token yet)
    token = await _create_verify_token(user["_id"])
    verify_url = _verify_link(token)
    try:
        send_verification_email(email_lc, verify_url)
    except Exception:
        # if email sending fails, you may want to delete user or allow resend
        # here we keep user and allow "resend" from UI
        pass

    return {
        "message": "Signup successful. Please check your email to verify your account."
    }

# ----------- Verify (JSON when fetched, Redirect when navigated) -----------

@router.get("/verify/{token}")
async def verify_path(token: str, request: Request):
    await _verify_token_and_mark_used(token)
    if _wants_json(request):
        return JSONResponse({"ok": True, "message": "Email verified successfully"})
    return RedirectResponse(url=f"{FRONTEND_BASE_URL}/login?verified=1", status_code=307)

@router.get("/verify")
async def verify_query(request: Request, token: str = Query(...), redirect: Optional[str] = Query(None)):
    await _verify_token_and_mark_used(token)
    if _wants_json(request, redirect_param=redirect):
        return JSONResponse({"ok": True, "message": "Email verified successfully"})
    return RedirectResponse(url=f"{FRONTEND_BASE_URL}/login?verified=1", status_code=307)

@router.post("/verify")
async def verify_post(payload: dict = Body(...)):
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="token required")
    await _verify_token_and_mark_used(token)
    return {"ok": True, "message": "Email verified successfully"}

# ----------- Resend verification -----------

@router.post("/resend-verification")
async def resend_verification(payload: dict = Body(...)):
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="email required")

    email_lc = str(email).strip().lower()  # ✅ normalize

    user = await db.users.find_one({"email": email_lc})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.get("is_verified"):
        return {"ok": True, "message": "Already verified"}

    token = await _create_verify_token(user["_id"])
    verify_url = _verify_link(token)
    try:
        send_verification_email(email_lc, verify_url)
    except Exception:
        pass
    return {"ok": True, "message": "Verification email sent"}

# ----------- Login -----------

@router.post("/login")
async def login(data: LoginRequest):
    email_lc = str(data.email).strip().lower()  # ✅ normalize
    user = await db.users.find_one({"email": email_lc})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not pwd_context.verify(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # block unverified users (your login page already handles 403 & shows resend button)
    if not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Email not verified")

    # ✅ add exp/iat claims so tokens expire (caught by get_current_user handler)
    now = _now_utc()
    token = jwt.encode(
        {
            "email": user["email"],
            "id": str(user["_id"]),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "token": token,
        "message": "Login successful",
        "user": {
            "email": user["email"],
            "firstName": user.get("firstName", ""),
            "lastName": user.get("lastName", ""),
            "company": user.get("company", ""),
            "jobTitle": user.get("jobTitle", "")
        }
    }

# ----------- Auth Dependency (unchanged) -----------

async def get_current_user(request: Request):
    """
    Dependency to extract and validate JWT from Authorization header or cookies.
    Returns user object with `.id` and `.email`.
    """
    token: Optional[str] = None

    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]

    # Try cookies
    if not token:
        token = request.cookies.get("token")

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("id")
        email = payload.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # Return as a simple object with `.id` and `.email`
        return SimpleNamespace(id=str(user_id), email=email)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
