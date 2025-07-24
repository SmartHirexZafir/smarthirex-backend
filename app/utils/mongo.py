# app/utils/mongo.py

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import certifi

load_dotenv()  # ✅ Load variables from .env

MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB_NAME]
