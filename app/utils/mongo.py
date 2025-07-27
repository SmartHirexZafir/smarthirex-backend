# ✅ File: app/utils/mongo.py

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import certifi

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
