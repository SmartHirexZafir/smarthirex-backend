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
# Existing helpers (UNCHANGED behavior, with additive ownership support)
# -----------------------------------------------------------------------------

# ✅ Optional: Called at app startup to confirm DB
async def verify_mongo_connection():
    try:
        # Lightweight ping first (fast fail if network/creds are wrong)
        await db.command("ping")
        collections = await db.list_collection_names()
        print(f"✅ MongoDB connected to '{MONGO_DB_NAME}'. Collections: {collections}")
        # ✅ Indexes (idempotent)
        await ensure_users_indexes()  # ← added for Google Login email/google_id lookups
        await ensure_email_verification_indexes()  # ← added to support verification flow
        await ensure_meetings_indexes()
        await ensure_parsed_resumes_indexes()
        await ensure_chat_queries_indexes()
        await ensure_search_history_indexes()
        await ensure_token_blacklist_indexes()  # ✅ NEW: token blacklist indexes
        await ensure_tests_indexes()  # ✅ NEW: tests/invites/submissions safety indexes
    except Exception as e:
        print("❌ MongoDB connection failed:", str(e))
        raise


# ✅ NEW: small projection helper to avoid pulling heavy fields in list views
def resume_lean_projection(extra_excludes: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Returns a projection dict that excludes heavy fields by default.
    Usage in finds: db.parsed_resumes.find(query, resume_lean_projection())
    """
    proj = {
        "raw_text": 0,          # often large
        "resume": 0,            # nested parsed payload (large)
        "embedding": 0,         # large vector
        "content_hash": 0,      # not needed in listings
    }
    if extra_excludes:
        proj.update(extra_excludes)
    return proj


# ✅ NEW: Sanitize documents before insert/update to avoid null proliferation
def sanitize_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    For known fields only; do not mutate large blobs unnecessarily.
    Replaces None with sensible defaults so frontend can rely on shapes.
    """
    out = dict(d)

    defaults = {
        "name": "",
        "email": "",
        "phone": "",
        "predicted_role": "",
        "role_norm": "",
        "location": "",
        "experience": 0,
        "yoe_num": 0,
        "resume_url": "",
        "company": "",
        "category": "",
        "raw_text": "",
    }
    for k, v in defaults.items():
        if out.get(k) is None:
            out[k] = v

    # list defaults
    if out.get("skills") is None or not isinstance(out.get("skills"), list):
        out["skills"] = []
    if out.get("skills_norm") is None or not isinstance(out.get("skills_norm"), list):
        out["skills_norm"] = []
    if out.get("education") is None or not isinstance(out.get("education"), list):
        out["education"] = []
    if out.get("projects") is None or not isinstance(out.get("projects"), list):
        out["projects"] = []

    # Normalize nested resume
    r = out.get("resume") or {}
    out["resume"] = {
        "summary": r.get("summary") or "",
        "education": r.get("education") or [],
        "workHistory": r.get("workHistory") or [],
        "projects": r.get("projects") or [],
        "filename": r.get("filename") or out.get("filename") or "resume.pdf",
        "url": r.get("url") or out.get("resume_url") or "",
    }

    return out


# ✅ NEW: Compute SHA256 hash of resume content for duplicate detection
def compute_cv_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ✅ UPDATED: Check if a resume already exists in DB by hash (scoped per user if provided)
async def check_duplicate_cv_hash(text: str, owner_user_id: Optional[str] = None) -> bool:
    """
    If owner_user_id is provided, duplicate check is scoped to that user.
    If not provided, falls back to legacy global check (backward-compatible).
    """
    hash_val = compute_cv_hash(text)
    query: Dict[str, Any] = {"content_hash": hash_val}
    if owner_user_id:
        query["ownerUserId"] = str(owner_user_id)
    existing = await db.parsed_resumes.find_one(query)
    return existing is not None


# ✅ UPDATED: Insert if not duplicate (returns inserted_id or None); supports ownership
async def insert_unique_resume(doc: dict, owner_user_id: Optional[str] = None) -> Optional[str]:
    """
    If owner_user_id is provided, the doc is attributed to that user and
    duplicate detection is performed per-user. Otherwise retains legacy behavior.
    """
    text = doc.get("raw_text", "")
    doc["content_hash"] = compute_cv_hash(text)

    if owner_user_id:
        doc["ownerUserId"] = str(owner_user_id)

    if await check_duplicate_cv_hash(text, owner_user_id=owner_user_id):
        print("⚠️ Duplicate resume detected. Skipping insert.")
        return None

    result = await db.parsed_resumes.insert_one(doc)
    return str(result.inserted_id)


# -----------------------------------------------------------------------------
# ✅ Index helpers
# -----------------------------------------------------------------------------

async def ensure_users_indexes() -> None:
    """
    Helpful indexes for the `users` collection.
    Needed for fast lookups and uniqueness with Google Login.
    Safe to call multiple times.
    """
    try:
        # Primary unique identity
        await db.users.create_index("email", unique=True)
        # Google account linkage
        await db.users.create_index("google_id")
        await db.users.create_index([("auth_provider", 1)])
        # Common query paths
        await db.users.create_index([("created_at", -1)])
        await db.users.create_index([("updated_at", -1)])
    except Exception as e:
        print("⚠️ Failed to create users indexes:", e)


async def ensure_email_verification_indexes() -> None:
    """
    Indexes for email_verification_tokens used during signup verification.
    """
    try:
        await db.email_verification_tokens.create_index("token", unique=True)
        await db.email_verification_tokens.create_index([("user_id", 1)])
        await db.email_verification_tokens.create_index([("expires_at", 1)])
        await db.email_verification_tokens.create_index([("used", 1)])
    except Exception as e:
        print("⚠️ Failed to create email_verification_tokens indexes:", e)


async def ensure_meetings_indexes() -> None:
    """
    Create helpful indexes for the `meetings` collection.
    Safe to call multiple times.
    """
    try:
        await db.meetings.create_index("candidate_id")
        await db.meetings.create_index([("starts_at", 1)])
        await db.meetings.create_index("token", unique=True)
        # Optional but useful if you added ownership to meetings:
        await db.meetings.create_index("ownerUserId")
    except Exception as e:  # don't fail app startup if index creation fails
        print("⚠️ Failed to create meetings indexes:", e)


async def ensure_parsed_resumes_indexes() -> None:
    """
    Helpful indexes for isolation & performance on parsed_resumes.
    Safe to call multiple times.

    This includes:
    - Ownership scoping
    - Duplicate detection (content_hash)
    - Legacy experience indexes (kept for backward compatibility)
    - ✅ New normalized-field indexes for fast exact filtering:
        role_norm, yoe_num, skills_norm (multikey), projects_norm (multikey)
    - ✅ Embedding lifecycle observability: embedding_status
    - ✅ Composite 'common query paths' indexes
    """
    try:
        # Ownership & duplicate detection
        await db.parsed_resumes.create_index("ownerUserId")
        await db.parsed_resumes.create_index("content_hash")
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("content_hash", 1)])

        # Legacy role/category (kept)
        await db.parsed_resumes.create_index([("predicted_role", 1)])
        await db.parsed_resumes.create_index([("category", 1)])

        # Legacy experience variants (kept to avoid breaking existing queries)
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("total_experience_years", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("years_of_experience", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("experience_years", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("yoe", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("experience", 1)])

        # ✅ New: normalized, canonical fields for fast exact queries
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("role_norm", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("yoe_num", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("skills_norm", 1)])     # multikey
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("projects_norm", 1)])   # multikey

        # ✅ New: observability for embedding pipeline
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("embedding_status", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("embedding_model_version", 1)])

        # ✅ Optimized composite indexes for category-first filtering (priority order)
        # Role-first indexes (most common filter)
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("role_norm", 1), ("yoe_num", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("role_norm", 1), ("skills_norm", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("role_norm", 1), ("location_norm", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("predicted_role", 1), ("experience", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("category", 1), ("experience", 1)])
        # Location index for location filtering
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("location_norm", 1)])
        # Skills index for skills filtering
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("skills", 1)])  # multikey
        # Education indexes
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("schools_detected", 1)])
        await db.parsed_resumes.create_index([("ownerUserId", 1), ("degrees_detected", 1)])

    except Exception as e:
        print("⚠️ Failed to create parsed_resumes indexes:", e)


# ✅ Indexes for chatbot logs (used by chatbot_router)
async def ensure_chat_queries_indexes() -> None:
    """
    Indexes for chat_queries, which stores per-owner query logs.
    Speeds up analytics and owner's history lookups.
    """
    try:
        await db.chat_queries.create_index([("ownerUserId", 1), ("timestamp", -1)])
        await db.chat_queries.create_index([("timestamp", -1)])
    except Exception as e:
        print("⚠️ Failed to create chat_queries indexes:", e)


# ✅ Indexes for search_history (used to render prior filtered results fast)
async def ensure_search_history_indexes() -> None:
    """
    Indexes for search_history collection.
    """
    try:
        await db.search_history.create_index([("ownerUserId", 1), ("timestamp_raw", -1)])
        await db.search_history.create_index([("timestamp_raw", -1)])
        await db.search_history.create_index([("totalMatches", -1)])  # ✅ for "Most Matches" sort
        await db.search_history.create_index([("prompt", 1)])         # ✅ for regex filtering
    except Exception as e:
        print("⚠️ Failed to create search_history indexes:", e)


# ✅ NEW: token blacklist indexes (for auth/session revocation)
async def ensure_token_blacklist_indexes() -> None:
    """
    Indexes for token_blacklist collection.
    """
    try:
        await db.token_blacklist.create_index("jti", unique=True)
        await db.token_blacklist.create_index("expires_at", expireAfterSeconds=0)
        await db.token_blacklist.create_index([("user_id", 1), ("revoked_at", -1)])
    except Exception as e:
        print("⚠️ Failed to create token_blacklist indexes:", e)


async def ensure_tests_indexes() -> None:
    """
    Indexes for Smart AI/custom test lifecycle collections.
    Kept additive and backward-compatible.
    """
    await db.test_invites.create_index("token", unique=True)
    await db.test_invites.create_index([("candidateId", 1), ("status", 1)])
    await db.test_invites.create_index([("createdAt", -1)])
    await db.test_invites.create_index([("expiresAt", 1)])
    await db.test_invites.create_index(
        [("candidateId", 1)],
        unique=True,
        name="uniq_candidate_active_invite",
        partialFilterExpression={"status": {"$in": ["pending", "active"]}},
    )

    await db.tests.create_index("token", unique=True)
    await db.tests.create_index([("candidateId", 1), ("status", 1)])
    await db.tests.create_index([("startedAt", -1)])
    await db.tests.create_index([("expiresAt", 1)])
    await db.tests.create_index(
        [("candidateId", 1)],
        unique=True,
        name="uniq_candidate_open_test",
        partialFilterExpression={"status": {"$in": ["active", "submitted"]}},
    )

    await db.test_submissions.create_index("testId", unique=True)
    await db.test_submissions.create_index([("candidateId", 1), ("submittedAt", -1)])
    await db.test_submissions.create_index([("needs_marking", 1), ("submittedAt", -1)])
    try:
        await db.test_submissions.create_index("candidateId", unique=True, name="uniq_candidate_single_attempt")
    except Exception as idx_err:
        dupes = []
        try:
            cursor = db.test_submissions.aggregate([
                {"$group": {"_id": "$candidateId", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}},
                {"$limit": 10},
            ])
            dupes = [str(d.get("_id")) async for d in cursor]
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to enforce unique one-attempt-per-candidate index on test_submissions.candidateId. "
            f"Resolve duplicate legacy data first. Sample duplicate candidateIds: {dupes}. "
            f"Original error: {idx_err}"
        ) from idx_err


# -----------------------------------------------------------------------------
# ✅ Meetings helpers for Schedule feature (non-breaking additions)
# -----------------------------------------------------------------------------

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


async def list_meetings_by_candidate(candidate_id: str, owner_user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List meetings for a given candidate, most recent first.
    If owner_user_id is provided, results are scoped to that owner (recommended).
    """
    query: Dict[str, Any] = {"candidate_id": candidate_id}
    if owner_user_id:
        query["ownerUserId"] = str(owner_user_id)

    out: List[Dict[str, Any]] = []
    cursor = db.meetings.find(query).sort("starts_at", -1)
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
