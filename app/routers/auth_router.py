# app/routers/auth_router.py

from fastapi import APIRouter, HTTPException, Request, status, Body, Query, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from app.utils.mongo import db
from app.utils.emailer import send_verification_email
from app.utils.datetime_serialization import serialize_utc
from dotenv import load_dotenv
from types import SimpleNamespace
from typing import Optional, Dict
from datetime import datetime, timedelta, timezone  # ✅ include timezone for aware datetimes
import jwt, uuid, os, json
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen, Request as UrlRequest

load_dotenv()

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET must be set in environment")
ALGORITHM = "HS256"
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000").rstrip("/")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:10000").rstrip("/")
VERIFICATION_TTL_HOURS = 24
JWT_EXPIRES_HOURS = int(os.getenv("JWT_EXPIRES_HOURS", "168"))  # ✅ default 7 days

# --- Google OAuth Settings ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"  # token verification endpoint

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

class CompleteProfileRequest(BaseModel):
    name: str
    role: str
    company: str

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

# ✅ NEW: normalize any datetime (or ISO string) to aware UTC for safe comparisons
def _to_aware_utc(dt) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    # If it's neither str nor datetime, raise a clear error
    raise HTTPException(status_code=500, detail="Invalid datetime format in token")

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
    # ✅ CHANGE: normalize stored expires_at to aware UTC before comparing
    exp = _to_aware_utc(tok["expires_at"])
    if exp < _now_utc():  # ✅ aware comparison
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

def _app_jwt_for_user(user: Dict) -> str:
    """Issue our application JWT for a user record (with unique jti)."""
    now = _now_utc()
    jti = str(uuid.uuid4())  # ✅ unique token id for revocation
    return jwt.encode(
        {
            "email": user["email"],
            "id": str(user["_id"]),
            "jti": jti,  # ✅ include jti
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )

def _iso(dt: Optional[datetime]) -> str:
    if isinstance(dt, datetime):
        try:
            return serialize_utc(dt)
        except Exception:
            return ""
    return ""

def _coalesce_str(v) -> str:
    return v if isinstance(v, str) else ("" if v is None else str(v))

def _serialize_user_for_client(user: Dict) -> Dict:
    """
    Return a client-safe user payload with safe defaults (no nulls).
    Includes common avatar fields and timestamps as ISO strings.
    """
    first = _coalesce_str(user.get("firstName", ""))
    last = _coalesce_str(user.get("lastName", ""))
    display_name = (first + " " + last).strip() or _coalesce_str(user.get("displayName", ""))
    role = _coalesce_str(user.get("jobTitle", "") or user.get("role", ""))
    company = _coalesce_str(user.get("company", ""))
    email = _coalesce_str(user.get("email", ""))

    # avatar fields harmonized
    avatar_url = (
        _coalesce_str(user.get("avatarUrl"))
        or _coalesce_str(user.get("avatar_url"))
        or _coalesce_str(user.get("avatar"))
        or _coalesce_str(user.get("photoUrl"))
        or _coalesce_str(user.get("photo_url"))
    )
    created_at = user.get("created_at")
    updated_at = user.get("updated_at")

    return {
        "id": _coalesce_str(user.get("_id")),
        "email": email,
        "firstName": first,
        "lastName": last,
        "company": company,
        "jobTitle": _coalesce_str(user.get("jobTitle", "")),
        "role": role,
        "name": display_name,
        "authProvider": _coalesce_str(user.get("auth_provider", "password")),
        # avatar variants (frontend will pick what it needs)
        "avatar": avatar_url,          # generic
        "avatarUrl": avatar_url,       # camelCase
        "photoUrl": avatar_url,        # legacy
        # timestamps (ISO)
        "createdAt": _iso(created_at),
        "updatedAt": _iso(updated_at),
    }

def _split_name(full_name: str) -> Dict[str, str]:
    full_name = (full_name or "").strip()
    if not full_name:
        return {"firstName": "", "lastName": ""}
    parts = full_name.split()
    if len(parts) == 1:
        return {"firstName": parts[0], "lastName": ""}
    return {"firstName": " ".join(parts[:-1]), "lastName": parts[-1]}

# ----------- Auth Dependency (moved up so it exists before use) -----------

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
        jti = payload.get("jti")  # ✅ must be present
        if not user_id or not jti:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # ✅ Revocation check (blacklist)
        revoked = await db.token_blacklist.find_one({"jti": jti})
        if revoked:
            raise HTTPException(status_code=401, detail="Token revoked")

        # Return as a simple object with `.id` and `.email`
        return SimpleNamespace(id=str(user_id), email=email)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------- Signup -----------

@router.post("/signup")
async def signup(data: SignupRequest):
    # ✅ normalize email to lowercase to avoid case-variant duplicates
    email_lc = str(data.email).strip().lower()

    existing = await db.users.find_one({"email": email_lc})
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

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

    # ✅ add exp/iat claims AND jti so tokens are revocable
    now = _now_utc()
    jti = str(uuid.uuid4())
    token = jwt.encode(
        {
            "email": user["email"],
            "id": str(user["_id"]),
            "jti": jti,  # ✅ include jti
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    payload = {
        "token": token,
        "message": "Login successful",
        "user": _serialize_user_for_client(user),
    }

    secure_cookie = BACKEND_BASE_URL.startswith("https://")
    response = JSONResponse(payload)
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=JWT_EXPIRES_HOURS * 3600,
        path="/",
    )
    return response

@router.post("/logout")
async def logout(request: Request):
    """
    Real logout: revoke the current JWT by storing its jti in a blacklist
    (TTL via expires_at index). Always clears the cookie.
    """
    token: Optional[str] = None

    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]

    # Try cookie
    if not token:
        token = request.cookies.get("token")

    if token:
        try:
            # Decode but ignore expiration to capture claims for blacklisting if possible
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM],
                options={"verify_exp": False}
            )
            jti = payload.get("jti")
            uid = str(payload.get("id") or "")
            exp_ts = payload.get("exp")
            if jti:
                expires_at = (
                    datetime.fromtimestamp(exp_ts, tz=timezone.utc)
                    if isinstance(exp_ts, (int, float))
                    else _now_utc() + timedelta(hours=1)
                )
                # Upsert into blacklist
                await db.token_blacklist.update_one(
                    {"jti": jti},
                    {
                        "$set": {
                            "jti": jti,
                            "user_id": uid,
                            "expires_at": expires_at,
                            "revoked_at": _now_utc(),
                        }
                    },
                    upsert=True,
                )
        except Exception:
            # Swallow errors: we still clear the cookie and return ok
            pass

    # ✅ ensure cookie is cleared; keep payload simple and stable
    response = JSONResponse({"ok": True})
    response.delete_cookie("token", path="/")
    return response

# ----------- Me (used by Google flow / profile page to fetch current profile) -----------

@router.get("/me")
async def me(current: SimpleNamespace = Depends(get_current_user)):
    user = await db.users.find_one({"_id": current.id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # ✅ return complete, null-safe profile inside "user" for frontend compatibility
    return {"user": _serialize_user_for_client(user)}

# ----------- Complete profile (Google flow) -----------

@router.post("/complete-profile")
async def complete_profile(
    data: CompleteProfileRequest,
    current: SimpleNamespace = Depends(get_current_user),
):
    name_parts = _split_name(data.name)
    update_doc = {
        "firstName": name_parts["firstName"],
        "lastName": name_parts["lastName"],
        "jobTitle": data.role.strip(),
        "company": data.company.strip(),
        # Keep convenience displayName for header if your app uses it elsewhere
        "displayName": data.name.strip(),
        "updated_at": _now_utc(),
    }
    await db.users.update_one({"_id": current.id}, {"$set": update_doc})
    updated = await db.users.find_one({"_id": current.id})
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": _serialize_user_for_client(updated)}

# ----------- Google OAuth -----------

@router.get("/google")
async def google_login(redirect_url: Optional[str] = Query(None)):
    """
    Initiates Google OAuth2 (OpenID Connect). We pack the desired post-login redirect
    in a signed state token to protect against CSRF.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured on the server")

    # Default to frontend /login which your client expects to read token from query
    desired_redirect = (redirect_url or f"{FRONTEND_BASE_URL}/login").strip()

    # Signed state (short-lived)
    now = _now_utc()
    state = jwt.encode(
        {
            "r": desired_redirect,
            "nonce": str(uuid.uuid4()),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    callback_url = f"{BACKEND_BASE_URL}/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": callback_url,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "consent",
        "access_type": "offline",
        "include_granted_scopes": "true",
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=307)

@router.get("/google/callback")
async def google_callback(code: Optional[str] = Query(None), state: Optional[str] = Query(None)):
    """
    Handles Google's callback: exchanges code for tokens, validates id_token, and
    creates/links a user. Redirects back to the frontend with ?oauth=google&token=...&needs_profile=0|1
    """
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")

    try:
        state_payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
        desired_redirect: str = state_payload.get("r") or f"{FRONTEND_BASE_URL}/login"
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    # Exchange code for tokens
    callback_url = f"{BACKEND_BASE_URL}/auth/google/callback"
    token_req_body = urlencode({
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": callback_url,
        "grant_type": "authorization_code",
    }).encode("utf-8")

    try:
        req = UrlRequest(GOOGLE_TOKEN_URL, data=token_req_body, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        with urlopen(req) as resp:
            token_resp = json.loads(resp.read().decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code")

    id_token = token_resp.get("id_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="No id_token in token response")

    # Validate id_token via Google's tokeninfo
    try:
        with urlopen(f"{GOOGLE_TOKENINFO_URL}?{urlencode({'id_token': id_token})}") as r:
            tokeninfo = json.loads(r.read().decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to validate Google ID token")

    # Basic checks
    aud = tokeninfo.get("aud")
    if aud != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Google ID token audience mismatch")

    email = (tokeninfo.get("email") or "").strip().lower()
    sub = tokeninfo.get("sub")  # Google's user id
    email_verified = str(tokeninfo.get("email_verified", "false")).lower() == "true"
    profile_name = tokeninfo.get("name") or ""
    # picture = tokeninfo.get("picture")  # available if you want to store

    if not email or not sub:
        raise HTTPException(status_code=400, detail="Missing email or subject from Google")

    # Upsert user
    user = await db.users.find_one({"email": email})
    if user:
        # Link Google account if not linked yet
        update = {
            "google_id": sub,
            "auth_provider": "google",
            "is_verified": True if email_verified else True,  # treat as verified after Google
            "updated_at": _now_utc(),
        }
        # If no name is present, prefill from Google
        if not user.get("firstName") and not user.get("lastName") and profile_name:
            parts = _split_name(profile_name)
            update["firstName"] = parts["firstName"]
            update["lastName"] = parts["lastName"]
            update["displayName"] = profile_name
        await db.users.update_one({"_id": user["_id"]}, {"$set": update})
        user = await db.users.find_one({"_id": user["_id"]})
    else:
        # Create a new user record
        parts = _split_name(profile_name)
        user = {
            "_id": str(uuid.uuid4()),
            "email": email,
            "firstName": parts["firstName"],
            "lastName": parts["lastName"],
            "displayName": profile_name,
            "company": "",
            "jobTitle": "",
            "password": "",  # no password for Google accounts
            "is_verified": True if email_verified else True,
            "google_id": sub,
            "auth_provider": "google",
            "created_at": _now_utc(),
            "updated_at": _now_utc(),
        }
        await db.users.insert_one(user)

    # Issue our app token (helper includes jti)
    token = _app_jwt_for_user(user)

    # Determine if we need to ask for name/role/company (as requested)
    serialized = _serialize_user_for_client(user)
    has_name = bool(serialized.get("name"))
    has_role = bool(serialized.get("role"))
    has_company = bool(serialized.get("company"))
    needs_profile = not (has_name and has_role and has_company)

    # Redirect to frontend login page which will store token and show profile modal if needed
    final_redirect = f"{desired_redirect.split('?')[0]}?oauth=google&token={quote_plus(token)}&needs_profile={'1' if needs_profile else '0'}"

    # Set cookie as a convenience (client also reads the query param)
    response = RedirectResponse(url=final_redirect, status_code=307)
    secure_cookie = BACKEND_BASE_URL.startswith("https://")
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        max_age=JWT_EXPIRES_HOURS * 3600,
        path="/",
    )
    return response
