from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import certifi
import hashlib

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

# ✅ Optional: Called at app startup to confirm DB
async def verify_mongo_connection():
    try:
        collections = await db.list_collection_names()
        print(f"✅ MongoDB connected to '{MONGO_DB_NAME}'. Collections: {collections}")
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
async def insert_unique_resume(doc: dict) -> str:
    text = doc.get("raw_text", "")
    doc["content_hash"] = compute_cv_hash(text)

    if await check_duplicate_cv_hash(text):
        print("⚠️ Duplicate resume detected. Skipping insert.")
        return None

    result = await db.parsed_resumes.insert_one(doc)
    return result.inserted_id
