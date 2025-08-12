# app/utils/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import certifi
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

# For working with _id when needed
try:
    from bson import ObjectId
except Exception:  # pragma: no cover
    ObjectId = None  # type: ignore

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Load Mongo URI and DB name safely
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# ✅ Fail clearly if missing
if not MONGODB_URI:
    raise RuntimeError("❌ MONGODB_URI not set in .env")

if not MONGO_DB_NAME:
    raise RuntimeError("❌ MONGO_DB_NAME not set in .env")

# ✅ Create Mongo client and DB
client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB_NAME]

# -----------------------------------------------------------------------------
# Existing helpers (UNCHANGED behavior)
# -----------------------------------------------------------------------------

# ✅ Optional: Called at app startup to confirm DB
async def verify_mongo_connection():
    try:
        collections = await db.list_collection_names()
        print(f"✅ MongoDB connected to '{MONGO_DB_NAME}'. Collections: {collections}")
        # ✅ NEW (non-breaking): ensure useful indexes for meetings collection
        await ensure_meetings_indexes()
    except Exception as e:
        print("❌ MongoDB connection failed:", str(e))
        raise


# ✅ NEW: Compute SHA256 hash of resume content for duplicate detection
def compute_cv_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ✅ NEW: Check if a resume already exists in DB by hash
async def check_duplicate_cv_hash(text: str) -> bool:
    hash_val = compute_cv_hash(text)
    existing = await db.parsed_resumes.find_one({"content_hash": hash_val})
    return existing is not None


# ✅ NEW: Insert if not duplicate (returns inserted_id or None)
async def insert_unique_resume(doc: dict) -> Optional[str]:
    text = doc.get("raw_text", "")
    doc["content_hash"] = compute_cv_hash(text)

    if await check_duplicate_cv_hash(text):
        print("⚠️ Duplicate resume detected. Skipping insert.")
        return None

    result = await db.parsed_resumes.insert_one(doc)
    return str(result.inserted_id)


# -----------------------------------------------------------------------------
# ✅ NEW: Meetings helpers for Schedule feature (non-breaking additions)
# -----------------------------------------------------------------------------

async def ensure_meetings_indexes() -> None:
    """
    Create helpful indexes for the `meetings` collection.
    Safe to call multiple times.
    """
    try:
        await db.meetings.create_index("candidate_id")
        await db.meetings.create_index([("starts_at", 1)])
        await db.meetings.create_index("token", unique=True)
    except Exception as e:  # don't fail app startup if index creation fails
        print("⚠️ Failed to create meetings indexes:", e)


async def create_meeting(doc: Dict[str, Any]) -> str:
    """
    Insert a meeting document. Returns inserted id as string.
    Caller is responsible for providing validated fields, e.g.:
      {
        candidate_id, email, starts_at (UTC aware datetime),
        timezone, duration_mins, title, notes?, status, token, meeting_url, created_at
      }
    """
    res = await db.meetings.insert_one(doc)
    return str(res.inserted_id)


async def get_meeting_by_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Find a meeting by its unique token.
    """
    return await db.meetings.find_one({"token": token})


async def list_meetings_by_candidate(candidate_id: str) -> List[Dict[str, Any]]:
    """
    List meetings for a given candidate, most recent first.
    """
    out: List[Dict[str, Any]] = []
    cursor = db.meetings.find({"candidate_id": candidate_id}).sort("starts_at", -1)
    async for m in cursor:
        m["_id"] = str(m.get("_id"))
        out.append(m)
    return out


async def mark_meeting_email_sent_by_id(meeting_id: str, *, sent: bool, error: Optional[str] = None) -> int:
    """
    Update email_sent fields by meeting _id (string).
    Returns modified_count.
    """
    if not ObjectId:
        return 0
    try:
        oid = ObjectId(meeting_id)
    except Exception:
        return 0

    update: Dict[str, Any] = {
        "$set": {
            "email_sent": bool(sent),
            "email_sent_at": datetime.now(timezone.utc) if sent else None,
        }
    }
    if not sent:
        update["$set"]["email_error"] = error or "unknown_error"

    res = await db.meetings.update_one({"_id": oid}, update)
    return res.modified_count


async def mark_meeting_email_sent_by_token(token: str, *, sent: bool, error: Optional[str] = None) -> int:
    """
    Update email_sent fields by meeting token.
    Returns modified_count.
    """
    update: Dict[str, Any] = {
        "$set": {
            "email_sent": bool(sent),
            "email_sent_at": datetime.now(timezone.utc) if sent else None,
        }
    }
    if not sent:
        update["$set"]["email_error"] = error or "unknown_error"

    res = await db.meetings.update_one({"token": token}, update)
    return res.modified_count
