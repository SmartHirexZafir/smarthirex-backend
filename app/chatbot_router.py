# ✅ File: app/chatbot_router.py

from fastapi import APIRouter, Request
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db
from datetime import datetime

router = APIRouter()

@router.post("/query")
async def handle_chatbot_query(request: Request):
    """
    Receives a natural language prompt from user,
    identifies intent, processes it, and responds accordingly.
    """
    data = await request.json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return {"message": "Prompt is empty."}

    # Step 1: Parse the prompt and detect intent
    parsed = parse_prompt(prompt)

    # Step 2: Get bot response
    response = await build_response(parsed)

    # Step 3: Log analytics (always)
    await db.chat_queries.insert_one({
        "prompt": prompt,
        "parsed": parsed,
        "response": response,
        "timestamp": datetime.utcnow()
    })

    # Step 4: Save history if this was a CV filtering prompt
    if parsed.get("intent") == "filter_cv" and response.get("resumes_preview"):
        timestamp_raw = datetime.utcnow()
        timestamp_display = datetime.now().strftime("%B %d, %Y – %I:%M %p")

        await db.search_history.insert_one({
            "prompt": prompt,
            "parsed": parsed,
            "timestamp_raw": timestamp_raw,
            "timestamp_display": timestamp_display,
            "totalMatches": len(response["resumes_preview"]),
            "candidates": response["resumes_preview"]
        })

    return response
