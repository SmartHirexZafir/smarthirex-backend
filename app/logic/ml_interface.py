import joblib
import os
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from app.utils.mongo import db
from sentence_transformers import SentenceTransformer, util

# Load models
MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

model: BaseEstimator = joblib.load(MODEL_PATH)
vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# Load semantic model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Clean text utility
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\S+@\S+", '', text)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# ✅ Final: Strict hybrid matching with safe thresholding
async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
    cleaned_prompt = clean_text(prompt)
    prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)

    is_prompt_role_like = len(cleaned_prompt.split()) <= 4
    matched = []
    seen_ids = set()

    cursor = db.parsed_resumes.find({})

    async for cv in cursor:
        cv_id = cv.get("_id")
        if not cv_id or cv_id in seen_ids:
            continue
        seen_ids.add(cv_id)

        raw_text = cv.get("raw_text", "")
        if not raw_text.strip():
            continue

        predicted_role = (cv.get("predicted_role") or "").lower()
        confidence = round(cv.get("confidence", 0), 2)
        cleaned_resume_text = clean_text(raw_text)

        # Decide what to compare with the prompt
        compare_text = predicted_role if is_prompt_role_like else cleaned_resume_text
        compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)

        similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
        semantic_score = round(similarity * 100, 2)

        # ✅ Enforce stricter semantic match for role-like prompts
        if is_prompt_role_like:
            if semantic_score < 70:
                continue  # Skip irrelevant roles

        # ✅ For content prompts, apply base threshold
        if not is_prompt_role_like:
            if semantic_score < threshold:
                continue

        # ✅ Score logic
        if confidence >= semantic_score:
            final_score = confidence
            score_type = "Model Score"
        else:
            final_score = semantic_score
            score_type = "Semantic Score"

        matched.append({
            "_id": cv_id,
            "name": cv.get("name") or "Unnamed",
            "predicted_role": cv.get("predicted_role") or "",
            "experience": cv.get("experience") or 0,
            "location": cv.get("location") or "N/A",
            "email": cv.get("email", "N/A"),
            "skills": cv.get("skills", []),
            "resume_url": cv.get("resume_url", ""),
            "semantic_score": semantic_score,
            "confidence": confidence,
            "score_type": score_type,
            "final_score": final_score,
            "rank": 0,
            "raw_text": raw_text
        })

    matched.sort(key=lambda x: x["final_score"], reverse=True)

    for i, cv in enumerate(matched):
        cv["rank"] = i + 1

    return matched
