# 📄 app/routers/history_router.py

from fastapi import APIRouter, Query, HTTPException
from bson import ObjectId
from datetime import datetime
from typing import Optional, List
from app.utils.mongo import db

router = APIRouter()

# Utility to convert ObjectId to str
def serialize_doc(doc):
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc

@router.get("/user-history")
async def get_history(
    dateFrom: Optional[str] = Query(None),
    dateTo: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: Optional[str] = Query("latest")
):
    query = {}

    if search:
        query["prompt"] = {"$regex": search, "$options": "i"}

    if dateFrom or dateTo:
        time_filter = {}
        if dateFrom:
            time_filter["$gte"] = datetime.fromisoformat(dateFrom)
        if dateTo:
            time_filter["$lte"] = datetime.fromisoformat(dateTo)
        query["timestamp_raw"] = time_filter

    sort_key = "timestamp_raw"
    sort_order = -1 if sort == "latest" else 1
    if sort == "mostMatches":
        sort_key = "totalMatches"

    cursor = db.search_history.find(query).sort(sort_key, sort_order)
    results = []
    async for doc in cursor:
        doc["timestamp"] = doc["timestamp_display"]
        results.append(serialize_doc(doc))
    return results

@router.get("/history-result/{history_id}")
async def get_history_result(history_id: str):
    doc = await db.search_history.find_one({"_id": ObjectId(history_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="History entry not found")
    doc["timestamp"] = doc["timestamp_display"]
    return serialize_doc(doc)
