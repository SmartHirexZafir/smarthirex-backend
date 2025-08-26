# ✅ File: app/chatbot_router.py

from fastapi import APIRouter, Request, Depends
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db
from datetime import datetime
from app.routers.auth_router import get_current_user  # ✅ enforce auth

router = APIRouter()

@router.post("/query")
async def handle_chatbot_query(
    request: Request,
    current_user = Depends(get_current_user),
):
    """
    Receives a natural language prompt from user,
    identifies intent, processes it, and responds accordingly.
    """
    data = await request.json()
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return {"message": "Prompt is empty.", "resumes_preview": []}

    owner_id = str(current_user.id)

    # ✅ Early guard: if this user has no CVs, short-circuit with a clear flag
    user_cv_count = await db.parsed_resumes.count_documents({"ownerUserId": owner_id})
    if user_cv_count == 0:
        return {
            "message": "no_cvs_uploaded",
            "no_cvs_uploaded": True,
            "resumes_preview": []
        }

    # Step 1: Parse intent
    parsed = parse_prompt(prompt)

    # Step 2: Build response (⚠️ without extra kwargs to avoid TypeError)
    response = await build_response(parsed)

    # ✅ Safety post-filter: keep only resumes owned by current user
    raw_preview = response.get("resumes_preview", []) or []
    filtered_preview = []
    for item in raw_preview:
        cid = item.get("_id") or item.get("id")
        if not cid:
            continue
        doc = await db.parsed_resumes.find_one(
            {"_id": cid, "ownerUserId": owner_id},
            {"_id": 1}
        )
        if doc:
            filtered_preview.append(item)

    # Step 3: Log analytics (scoped)
    await db.chat_queries.insert_one({
        "prompt": prompt,
        "parsed": parsed,
        "response_preview_count": len(filtered_preview),
        "timestamp": datetime.utcnow(),
        "ownerUserId": owner_id,
    })

    # Step 4: Save history for filter intents
    if parsed.get("intent") == "filter_cv" and filtered_preview:
        timestamp_raw = datetime.utcnow()
        timestamp_display = datetime.now().strftime("%B %d, %Y – %I:%M %p")
        await db.search_history.insert_one({
            "prompt": prompt,
            "parsed": parsed,
            "timestamp_raw": timestamp_raw,
            "timestamp_display": timestamp_display,
            "totalMatches": len(filtered_preview),
            "candidates": filtered_preview,
            "ownerUserId": owner_id,
        })

    # Step 5: Return
    return {
        "reply": response.get("reply", ""),
        "resumes_preview": filtered_preview
    }
