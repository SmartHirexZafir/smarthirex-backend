# ✅ File: app/logic/ml_interface.py

import joblib
from app.utils.mongo import db
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import os
import re

# Load model and preprocessing artifacts
MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

model: BaseEstimator = joblib.load(MODEL_PATH)
vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# Text cleaner (same as in model training)
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

# Main async function to rank resumes
async def get_ranked_resumes(filters: dict) -> List[Dict]:
    required_skills = set(s.lower() for s in filters.get("skills", []))
    min_experience = int(filters.get("experience") or 0)
    required_projects = set(p.lower() for p in filters.get("project_keywords", []))

    job_role_raw = filters.get("job_role")
    required_role = job_role_raw.lower() if isinstance(job_role_raw, str) else ""

    cursor = db.parsed_resumes.find({})
    matched = []

    async for cv in cursor:
        raw_text = cv.get("raw_text", "")
        experience = int(cv.get("experience") or 0)
        skills = set(s.lower() for s in cv.get("skills", []))

        if not raw_text.strip():
            continue

        # Clean and vectorize for model prediction
        cleaned_text = clean_text(raw_text)
        features = vectorizer.transform([cleaned_text])
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0].lower()
        cv["predicted_role"] = predicted_label

        # Filtering conditions
        skill_match = required_skills.issubset(skills) if required_skills else True
        exp_match = experience >= min_experience
        project_match = all(proj in cleaned_text for proj in required_projects) if required_projects else True
        role_match = required_role in predicted_label if required_role else True

        if skill_match and exp_match and project_match and role_match:
            # Match score (based on skill overlap)
            if required_skills:
                overlap = required_skills.intersection(skills)
                match_score = round((len(overlap) / len(required_skills)) * 100, 2)
            else:
                match_score = 100.0  # if no skills given, assume full match

            cv["match_score"] = match_score
            matched.append(cv)

    # Sort by match_score descending
    matched.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return matched
