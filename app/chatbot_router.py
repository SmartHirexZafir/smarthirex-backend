# app/chatbot_router.py

from fastapi import APIRouter, Request
from app.logic.intent_parser import parse_prompt
from app.logic.response_builder import build_response
from app.utils.mongo import db  # MongoDB connection

router = APIRouter()

@router.post("/query")
async def handle_chatbot_query(request: Request):
    """
    Receives a natural language prompt from user,
    identifies intent, processes it, and responds accordingly.
    """
    data = await request.json()
    prompt = data.get("prompt", "")

    if not prompt.strip():
        return {"message": "Prompt is empty."}

    # Parse the intent, extract skills & experience
    parsed = parse_prompt(prompt)

    # Await the response first
    response = await build_response(parsed)

    # Save interaction to MongoDB
    await db.chat_queries.insert_one({
        "prompt": prompt,
        "parsed": parsed,
        "response": response
    })

    return response
