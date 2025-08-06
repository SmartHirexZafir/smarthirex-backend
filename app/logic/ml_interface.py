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

# ✅ Extended real-world roles (priority first)
extended_roles = [
    "AI Engineer", "ML Engineer", "NLP Engineer", "Deep Learning Engineer",
    "MLOps Engineer", "Data Analyst", "Data Engineer", "Computer Vision Engineer",
    "Research Scientist", "Machine Learning Engineer"
]

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

# ✅ Hybrid filtering with prompt + predicted role + related roles + strengths
async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
    cleaned_prompt = clean_text(prompt)
    prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)
    is_prompt_role_like = len(cleaned_prompt.split()) <= 4

    # Extract filters
    years_req = 0
    required_skills = []
    location_filter = ""
    project_required = False

    exp_patterns = [
        r"(?:minimum|at least|more than)\s*(\d+)\s*(?:years|yrs)",
        r"(\d+)\s*\+?\s*(?:years|yrs)"
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, cleaned_prompt)
        if match:
            years_req = int(match.group(1))
            if "more than" in cleaned_prompt:
                years_req += 1
            break

    loc_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]+)", cleaned_prompt)
    if loc_match:
        location_filter = loc_match.group(1).strip().lower()

    if any(kw in cleaned_prompt for kw in ["project", "worked on", "contributed to"]):
        project_required = True

    skill_keywords = ["react", "aws", "node", "excel", "django", "figma", "pandas", "tensorflow", "keras", "java", "python"]
    required_skills = [kw for kw in skill_keywords if kw in cleaned_prompt]

    all_roles = list(label_encoder.classes_)
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

        original_predicted_role = cv.get("predicted_role") or ""
        predicted_role = original_predicted_role.lower().strip()
        confidence = round(cv.get("confidence", 0), 2)
        cleaned_resume_text = clean_text(raw_text)

        skills = cv.get("skills", [])
        skills_text = " ".join(skills)
        projects = cv.get("projects", []) if isinstance(cv.get("projects"), list) else []
        projects_text = " ".join([p if isinstance(p, str) else str(p) for p in projects])
        experience = cv.get("experience", 0)
        experience_text = f"{experience} years experience"
        location_text = (cv.get("location") or "").lower()

        # Filtering based on prompt intent
        if years_req > 0 and experience < years_req:
            continue
        if required_skills and not any(skill.lower() in skills_text.lower() for skill in required_skills):
            continue
        if location_filter and location_filter not in location_text:
            continue
        if project_required and not any("project" in proj.lower() or len(proj) > 30 for proj in projects):
            continue

        compare_text = predicted_role if is_prompt_role_like else " ".join([
            predicted_role,
            cleaned_resume_text,
            skills_text,
            projects_text,
            experience_text,
            location_text
        ])

        compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
        similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
        semantic_score = round(similarity * 100, 2)

        if is_prompt_role_like and semantic_score < 70:
            continue
        if not is_prompt_role_like and semantic_score < 60:
            continue

        # ✅ Related roles (top 3)
        related_roles = []
        if original_predicted_role.strip():
            try:
                pred_embed = embedding_model.encode(original_predicted_role)
                scores = []
                full_roles = extended_roles + all_roles

                for role in full_roles:
                    role_clean = role.strip().lower()
                    if role_clean == predicted_role:
                        continue
                    role_embed = embedding_model.encode(role)
                    score = util.cos_sim(pred_embed, role_embed)[0][0].item()
                    scores.append({
                        "role": role,
                        "match": round(score * 100, 2)
                    })

                scores.sort(key=lambda x: x["match"], reverse=True)
                related_roles = scores[:3]
            except Exception as e:
                print(f"[!] Related role generation failed: {e}")

        # ✅ Strengths & Red Flags (simple heuristics)
        strengths = []
        redFlags = []

        if semantic_score >= 85:
            strengths.append("Very strong semantic match")
        elif semantic_score >= 70:
            strengths.append("Relevant to prompt")

        if predicted_role in cleaned_prompt:
            strengths.append("Predicted role matches prompt")

        if not skills:
            redFlags.append("No skills extracted from resume")
        if experience < years_req:
            redFlags.append("Insufficient experience")

        final_score = semantic_score
        score_type = "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content"

        matched.append({
            "_id": cv_id,
            "name": cv.get("name") or "Unnamed",
            "predicted_role": original_predicted_role,
            "experience": experience,
            "location": location_text or "N/A",
            "email": cv.get("email", "N/A"),
            "skills": skills,
            "resume_url": cv.get("resume_url", ""),
            "semantic_score": semantic_score,
            "confidence": confidence,
            "score_type": score_type,
            "final_score": final_score,
            "related_roles": related_roles,
            "rank": 0,
            "raw_text": raw_text,
            "strengths": strengths,
            "redFlags": redFlags,
            "matchedSkills": [s for s in skills if s.lower() in cleaned_prompt.lower()],
            "missingSkills": [s for s in required_skills if s not in skills]
        })

    matched.sort(key=lambda x: x["final_score"], reverse=True)
    for i, cv in enumerate(matched):
        cv["rank"] = i + 1

    return matched







# import joblib
# import os
# import re
# import json
# from typing import List, Dict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.base import BaseEstimator
# from sklearn.preprocessing import LabelEncoder
# from app.utils.mongo import db
# from sentence_transformers import SentenceTransformer, util

# # Load models
# MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
# VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
# ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

# model: BaseEstimator = joblib.load(MODEL_PATH)
# vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
# label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# # Load semantic model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Load dynamic semantic role map from file
# SEMANTIC_ROLE_MAP_PATH = os.path.join("app", "semantic_role_pool.json")
# try:
#     with open(SEMANTIC_ROLE_MAP_PATH, "r", encoding="utf-8") as f:
#         semantic_role_map = json.load(f)
# except Exception as e:
#     print(f"[!] Failed to load semantic role pool: {e}")
#     semantic_role_map = {}

# # Clean text utility
# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"\S+@\S+", '', text)
#     text = re.sub(r"@\w+|#", '', text)
#     text = re.sub(r"[^a-zA-Z\s]", ' ', text)
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# # ✅ Hybrid filtering with prompt + predicted role + related roles
# async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
#     cleaned_prompt = clean_text(prompt)
#     prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)
#     is_prompt_role_like = len(cleaned_prompt.split()) <= 4

#     # Extract filters
#     years_req = 0
#     required_skills = []
#     location_filter = ""
#     project_required = False

#     exp_patterns = [
#         r"(?:minimum|at least|more than)\s*(\d+)\s*(?:years|yrs)",
#         r"(\d+)\s*\+?\s*(?:years|yrs)"
#     ]
#     for pattern in exp_patterns:
#         match = re.search(pattern, cleaned_prompt)
#         if match:
#             years_req = int(match.group(1))
#             if "more than" in cleaned_prompt:
#                 years_req += 1
#             break

#     loc_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]+)", cleaned_prompt)
#     if loc_match:
#         location_filter = loc_match.group(1).strip().lower()

#     if any(kw in cleaned_prompt for kw in ["project", "worked on", "contributed to"]):
#         project_required = True

#     skill_keywords = ["react", "aws", "node", "excel", "django", "figma", "pandas", "tensorflow", "keras", "java", "python"]
#     required_skills = [kw for kw in skill_keywords if kw in cleaned_prompt]

#     all_roles = list(label_encoder.classes_)

#     matched = []
#     seen_ids = set()
#     cursor = db.parsed_resumes.find({})

#     async for cv in cursor:
#         cv_id = cv.get("_id")
#         if not cv_id or cv_id in seen_ids:
#             continue
#         seen_ids.add(cv_id)

#         raw_text = cv.get("raw_text", "")
#         if not raw_text.strip():
#             continue

#         original_predicted_role = cv.get("predicted_role") or ""
#         predicted_role = original_predicted_role.lower().strip()
#         confidence = round(cv.get("confidence", 0), 2)
#         cleaned_resume_text = clean_text(raw_text)

#         skills = cv.get("skills", [])
#         skills_text = " ".join(skills)
#         projects = cv.get("projects", []) if isinstance(cv.get("projects"), list) else []
#         projects_text = " ".join(projects)
#         experience = cv.get("experience", 0)
#         experience_text = f"{experience} years experience"
#         location_text = (cv.get("location") or "").lower()

#         if years_req > 0 and experience < years_req:
#             continue
#         if required_skills and not any(skill.lower() in skills_text.lower() for skill in required_skills):
#             continue
#         if location_filter and location_filter not in location_text:
#             continue
#         if project_required and not any("project" in proj.lower() or len(proj) > 30 for proj in projects):
#             continue

#         compare_text = predicted_role if is_prompt_role_like else " ".join([
#             predicted_role,
#             cleaned_resume_text,
#             skills_text,
#             projects_text,
#             experience_text,
#             location_text
#         ])

#         compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
#         similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
#         semantic_score = round(similarity * 100, 2)

#         if is_prompt_role_like and semantic_score < 70:
#             continue
#         if not is_prompt_role_like and semantic_score < 60:
#             continue

#         # ✅ Related role prediction from JSON map
#         related_roles = []
#         try:
#             related_roles = semantic_role_map.get(original_predicted_role, [])
#         except Exception as e:
#             print(f"[!] Related role lookup failed: {e}")

#         final_score = semantic_score
#         score_type = "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content"

#         matched.append({
#             "_id": cv_id,
#             "name": cv.get("name") or "Unnamed",
#             "predicted_role": original_predicted_role,
#             "experience": experience,
#             "location": location_text or "N/A",
#             "email": cv.get("email", "N/A"),
#             "skills": skills,
#             "resume_url": cv.get("resume_url", ""),
#             "semantic_score": semantic_score,
#             "confidence": confidence,
#             "score_type": score_type,
#             "final_score": final_score,
#             "related_roles": related_roles,
#             "rank": 0,
#             "raw_text": raw_text
#         })

#     matched.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, cv in enumerate(matched):
#         cv["rank"] = i + 1

#     return matched


# import joblib
# import os
# import re
# from typing import List, Dict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.base import BaseEstimator
# from sklearn.preprocessing import LabelEncoder
# from app.utils.mongo import db
# from sentence_transformers import SentenceTransformer, util

# # Load models
# MODEL_PATH = os.path.join("app", "ml_models", "Resume_Ensemble_Model.pkl")
# VECTORIZER_PATH = os.path.join("app", "ml_models", "Resume_Tfidf_Vectorizer.pkl")
# ENCODER_PATH = os.path.join("app", "ml_models", "Resume_LabelEncoder.pkl")

# model: BaseEstimator = joblib.load(MODEL_PATH)
# vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
# label_encoder: LabelEncoder = joblib.load(ENCODER_PATH)

# # Load semantic model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Extended real-world roles (beyond label encoder)
# extended_roles = [
#     "AI Engineer", "ML Engineer", "NLP Engineer", "Deep Learning Engineer",
#     "MLOps Engineer", "Data Analyst", "Data Engineer", "Computer Vision Engineer",
#     "Research Scientist", "Machine Learning Engineer"
# ]

# # Clean text utility
# def clean_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"\S+@\S+", '', text)
#     text = re.sub(r"@\w+|#", '', text)
#     text = re.sub(r"[^a-zA-Z\s]", ' ', text)
#     text = re.sub(r"\s+", ' ', text).strip()
#     return text

# # ✅ Hybrid filtering with prompt + predicted role + related roles
# async def get_semantic_matches(prompt: str, threshold: float = 0.45) -> List[Dict]:
#     cleaned_prompt = clean_text(prompt)
#     prompt_embedding = embedding_model.encode(cleaned_prompt, convert_to_tensor=True)
#     is_prompt_role_like = len(cleaned_prompt.split()) <= 4

#     # Extract filters
#     years_req = 0
#     required_skills = []
#     location_filter = ""
#     project_required = False

#     exp_patterns = [
#         r"(?:minimum|at least|more than)\s*(\d+)\s*(?:years|yrs)",
#         r"(\d+)\s*\+?\s*(?:years|yrs)"
#     ]
#     for pattern in exp_patterns:
#         match = re.search(pattern, cleaned_prompt)
#         if match:
#             years_req = int(match.group(1))
#             if "more than" in cleaned_prompt:
#                 years_req += 1
#             break

#     loc_match = re.search(r"(?:in|from|based in)\s+([a-zA-Z\s]+)", cleaned_prompt)
#     if loc_match:
#         location_filter = loc_match.group(1).strip().lower()

#     if any(kw in cleaned_prompt for kw in ["project", "worked on", "contributed to"]):
#         project_required = True

#     skill_keywords = ["react", "aws", "node", "excel", "django", "figma", "pandas", "tensorflow", "keras", "java", "python"]
#     required_skills = [kw for kw in skill_keywords if kw in cleaned_prompt]

#     all_roles = list(label_encoder.classes_)
#     matched = []
#     seen_ids = set()
#     cursor = db.parsed_resumes.find({})

#     async for cv in cursor:
#         cv_id = cv.get("_id")
#         if not cv_id or cv_id in seen_ids:
#             continue
#         seen_ids.add(cv_id)

#         raw_text = cv.get("raw_text", "")
#         if not raw_text.strip():
#             continue

#         original_predicted_role = cv.get("predicted_role") or ""
#         predicted_role = original_predicted_role.lower().strip()
#         confidence = round(cv.get("confidence", 0), 2)
#         cleaned_resume_text = clean_text(raw_text)

#         skills = cv.get("skills", [])
#         skills_text = " ".join(skills)
#         projects = cv.get("projects", []) if isinstance(cv.get("projects"), list) else []
#         projects_text = " ".join(projects)
#         experience = cv.get("experience", 0)
#         experience_text = f"{experience} years experience"
#         location_text = (cv.get("location") or "").lower()

#         if years_req > 0 and experience < years_req:
#             continue
#         if required_skills and not any(skill.lower() in skills_text.lower() for skill in required_skills):
#             continue
#         if location_filter and location_filter not in location_text:
#             continue
#         if project_required and not any("project" in proj.lower() or len(proj) > 30 for proj in projects):
#             continue

#         compare_text = predicted_role if is_prompt_role_like else " ".join([
#             predicted_role,
#             cleaned_resume_text,
#             skills_text,
#             projects_text,
#             experience_text,
#             location_text
#         ])

#         compare_embedding = embedding_model.encode(compare_text, convert_to_tensor=True)
#         similarity = util.cos_sim(prompt_embedding, compare_embedding)[0][0].item()
#         semantic_score = round(similarity * 100, 2)

#         if is_prompt_role_like and semantic_score < 70:
#             continue
#         if not is_prompt_role_like and semantic_score < 60:
#             continue

#         # ✅ Real-time cosine similarity-based related role generation (Top 3 only)
#         related_roles = []
#         if original_predicted_role.strip():
#             try:
#                 pred_embed = embedding_model.encode(original_predicted_role)
#                 scores = []

#                 full_roles = list(set(all_roles + extended_roles))

#                 for role in full_roles:
#                     role_clean = role.strip().lower()
#                     if role_clean == predicted_role:
#                         continue
#                     role_embed = embedding_model.encode(role)
#                     score = util.cos_sim(pred_embed, role_embed)[0][0].item()
#                     scores.append({
#                         "role": role,
#                         "match": round(score * 100, 2)
#                     })

#                 scores.sort(key=lambda x: x["match"], reverse=True)
#                 related_roles = scores[:3]
#             except Exception as e:
#                 print(f"[!] Real-time related role generation failed: {e}")

#         final_score = semantic_score
#         score_type = "Prompt Match on Role" if is_prompt_role_like else "Prompt Match on Content"

#         matched.append({
#             "_id": cv_id,
#             "name": cv.get("name") or "Unnamed",
#             "predicted_role": original_predicted_role,
#             "experience": experience,
#             "location": location_text or "N/A",
#             "email": cv.get("email", "N/A"),
#             "skills": skills,
#             "resume_url": cv.get("resume_url", ""),
#             "semantic_score": semantic_score,
#             "confidence": confidence,
#             "score_type": score_type,
#             "final_score": final_score,
#             "related_roles": related_roles,
#             "rank": 0,
#             "raw_text": raw_text
#         })

#     matched.sort(key=lambda x: x["final_score"], reverse=True)
#     for i, cv in enumerate(matched):
#         cv["rank"] = i + 1

#     return matched
