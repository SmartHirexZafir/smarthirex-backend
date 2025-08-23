# app/routers/auth_router.py

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from app.utils.mongo import db
from dotenv import load_dotenv
import jwt, uuid, os
from types import SimpleNamespace
from typing import Optional

load_dotenv()

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET", "smarthirex-secret")
ALGORITHM = "HS256"

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

# ----------- Signup -----------

@router.post("/signup")
async def signup(data: SignupRequest):
    existing = await db.users.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed = pwd_context.hash(data.password)
    user = data.dict()
    user["_id"] = str(uuid.uuid4())
    user["password"] = hashed

    await db.users.insert_one(user)

    token = jwt.encode(
        {"email": data.email, "id": user["_id"]},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "token": token,
        "message": "Signup successful",
        "user": {
            "email": data.email,
            "firstName": data.firstName,
            "lastName": data.lastName,
            "company": data.company,
            "jobTitle": data.jobTitle
        }
    }

# ----------- Login -----------

@router.post("/login")
async def login(data: LoginRequest):
    user = await db.users.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not pwd_context.verify(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = jwt.encode(
        {"email": user["email"], "id": str(user["_id"])},
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

# ----------- Auth Dependency (NEW) -----------

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
