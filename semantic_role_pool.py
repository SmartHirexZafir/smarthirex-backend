import joblib
import json
from sentence_transformers import SentenceTransformer, util

# ✅ Correct path
label_encoder = joblib.load("app/ml_models/Resume_LabelEncoder.pkl")
all_roles = list(label_encoder.classes_)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
role_embeddings = embedding_model.encode(all_roles)

semantic_role_pool = {}
for i, role in enumerate(all_roles):
    similarities = util.cos_sim(role_embeddings[i], role_embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)

    related = []
    for idx in sorted_indices:
        if idx == i:
            continue
        related.append({
            "role": all_roles[idx],
            "match": round(float(similarities[idx]) * 100, 2)
        })
        if len(related) == 5:
            break
    semantic_role_pool[role] = related

# ✅ Save JSON into app/
with open("app/semantic_role_pool.json", "w", encoding="utf-8") as f:
    json.dump(semantic_role_pool, f, indent=2)

print("✅ semantic_role_pool.json generated successfully.")
