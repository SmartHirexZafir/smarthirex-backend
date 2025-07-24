# app/routers/auth_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from app.utils.mongo import db
from dotenv import load_dotenv
import jwt, uuid, os

load_dotenv()

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET", "smarthirex-secret")

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
        algorithm="HS256"
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
        algorithm="HS256"
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
