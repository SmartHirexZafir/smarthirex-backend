# app/scripts/backfill_embedding.py
"""
Backfill compact ANN index blobs + embeddings for parsed resumes.

- Uses the SAME blob-building logic and model choices as ml_interface.py:
  * index_blob: built from {predicted_role/category, name, location, skills, projects, experience, raw_text[:N]}
  * embedding: sentence-transformers model (SENTENCE_MODEL or LOCAL_SENTENCE_MODEL_DIR)
- Stores results on each resume doc:
    index_blob: str
    index_embedding: [float, ...]           # float32 list
    index_embedding_dim: int
    index_embedding_model: str              # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    index_embedding_updated_at: datetime (UTC ISO)
    ann_ready: True                         # convenience flag

Usage examples:
  # process everything
  python -m app.scripts.backfill_embedding

  # only a single owner
  python -m app.scripts.backfill_embedding --owner <ownerUserId>

  # recompute everything even if it already has an embedding
  python -m app.scripts.backfill_embedding --recompute

  # dry run (no DB writes), limit first 200 docs, batches of 64
  python -m app.scripts.backfill_embedding --dry-run --limit 200 --batch-size 64

Environment:
  SENTENCE_MODEL                (default: sentence-transformers/all-MiniLM-L6-v2)
  LOCAL_SENTENCE_MODEL_DIR      (default: app/ml_models/all-MiniLM-L6-v2)
  EMB_BATCH_SIZE                (default: 128)
  INDEX_TRUNCATE_CHARS          (default: 4000)

Relies on:
  - app.utils.mongo.db for the Motor client (configured via .env with MONGODB_URI, MONGO_DB_NAME)
"""

import argparse
import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.utils.mongo import db  # Motor (async)
from bson import ObjectId

# SentenceTransformer loader (aligned with ml_interface.py)
from sentence_transformers import SentenceTransformer

# Optional numpy (we handle gracefully if missing)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # we'll still store embeddings as Python floats if numpy isn't available


# -------------------------
# Config (mirror ml_interface)
# -------------------------
MODEL_ID = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_MODEL_DIR = os.getenv(
    "LOCAL_SENTENCE_MODEL_DIR",
    os.path.join("app", "ml_models", "all-MiniLM-L6-v2"),
)
DEFAULT_EMB_BATCH = int(os.getenv("EMB_BATCH_SIZE", "128"))
INDEX_TRUNCATE_CHARS = int(os.getenv("INDEX_TRUNCATE_CHARS", "4000"))


# -------------------------
# Model loader (robust)
# -------------------------
def _load_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(MODEL_ID)
    except Exception:
        pass
    try:
        return SentenceTransformer(LOCAL_MODEL_DIR, local_files_only=True)
    except Exception:
        pass
    return SentenceTransformer(MODEL_ID, local_files_only=True)


# -------------------------
# Helpers (clone of ml_interface behavior)
# -------------------------
def _clean_text(text: Any) -> str:
    import re
    s = str(text or "")
    s = s.lower()
    s = re.sub(r"http\S+|www\S+|https\S+", "", s)
    s = re.sub(r"\S+@\S+", "", s)
    s = re.sub(r"@\w+|#", "", s)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_search_blob(cv: Dict[str, Any]) -> str:
    """
    EXACTLY the same fields/logic we use in ml_interface._build_search_blob().
    This guarantees ANN & stored embeddings match behavior.
    """
    parts: List[str] = []
    pr = str(cv.get("predicted_role") or cv.get("category") or "").strip()
    nm = str(cv.get("name") or "").strip()
    loc = str(cv.get("location") or "").strip()
    exp = (
        cv.get("total_experience_years")
        or cv.get("years_of_experience")
        or cv.get("experience_years")
        or cv.get("yoe")
        or cv.get("experience")
    )

    if pr:
        parts.append(pr)
    if nm:
        parts.append(nm)
    if loc:
        parts.append(loc)

    skills = cv.get("skills") or []
    if isinstance(skills, list) and skills:
        parts.append(" ".join([str(s) for s in skills if s]))

    projects = cv.get("projects") or []
    if isinstance(projects, list) and projects:
        proj_bits: List[str] = []
        for p in projects[:6]:
            if isinstance(p, str):
                proj_bits.append(p)
            elif isinstance(p, dict):
                proj_bits.append(str(p.get("name") or ""))
                proj_bits.append(str(p.get("description") or ""))  # keep both
        if proj_bits:
            parts.append(" ".join([x for x in proj_bits if x]))

    if exp:
        parts.append(str(exp))

    raw_text = str(cv.get("raw_text") or "")
    if raw_text:
        parts.append(raw_text[:INDEX_TRUNCATE_CHARS])  # cap

    return _clean_text(" ".join([p for p in parts if p]).strip())


def _chunkify(lst: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size))
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -------------------------
# Core backfill
# -------------------------
async def backfill_embeddings(
    *,
    owner: Optional[str],
    limit: Optional[int],
    batch_size: int,
    recompute: bool,
    dry_run: bool,
) -> None:
    model = _load_embedding_model()

    # Build base query
    query: Dict[str, Any] = {}
    if owner:
        query["ownerUserId"] = owner

    # If not recomputing, skip docs that already have index_embedding
    if not recompute:
        query["index_embedding"] = {"$exists": False}

    # Projection minimal set; we still want raw_text for the blob (truncated)
    projection = {
        "_id": 1,
        "ownerUserId": 1,
        "predicted_role": 1,
        "category": 1,
        "name": 1,
        "location": 1,
        "skills": 1,
        "projects": 1,
        "total_experience_years": 1,
        "years_of_experience": 1,
        "experience_years": 1,
        "yoe": 1,
        "experience": 1,
        "raw_text": 1,
    }

    cursor = db.parsed_resumes.find(query, projection)
    if limit:
        cursor = cursor.limit(int(limit))

    # Motor's to_list requires a concrete integer length (not None)
    fetch_len = int(limit) if limit else 1_000_000
    docs: List[Dict[str, Any]] = await cursor.to_list(length=fetch_len)
    total = len(docs)
    if total == 0:
        print("No documents to process (query returned 0).")
        return

    print(f"Found {total} resume(s) to process. Batch size={batch_size}. Dry-run={dry_run}. Recompute={recompute}.")

    # Build blobs up-front
    ids: List[ObjectId] = []
    blobs: List[str] = []
    for cv in docs:
        _id = cv.get("_id")
        ids.append(_id)  # keep as-is; we handle string/ObjectId at update time
        blobs.append(_build_search_blob(cv))

    # Encode in batches
    attempted = 0
    updated = 0
    for batch_ids, batch_blobs in zip(_chunkify(ids, batch_size), _chunkify(blobs, batch_size)):
        # Compute embeddings
        try:
            embeddings = model.encode(
                batch_blobs,
                convert_to_tensor=False,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            print(f"❌ Encoding failed for a batch: {e}")
            continue

        # Ensure float32 list per vector
        to_store: List[List[float]] = []
        if np is not None:
            try:
                arr = np.array(embeddings, dtype=np.float32)
                to_store = [vec.astype(np.float32).tolist() for vec in arr]
            except Exception:
                # fallback: list of python floats
                to_store = [
                    [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
                    for vec in embeddings
                ]
        else:
            to_store = [
                [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]
                for vec in embeddings
            ]

        # Persist batch
        for _id, blob, vec in zip(batch_ids, batch_blobs, to_store):
            attempted += 1
            update_doc = {
                "index_blob": blob,
                "index_embedding": vec,
                "index_embedding_dim": len(vec),
                "index_embedding_model": MODEL_ID,
                "index_embedding_updated_at": _now_iso(),
                "ann_ready": True,
            }
            if dry_run:
                print(f"[DRY-RUN] Would update _id={_id} with embedding_dim={len(vec)}")
                continue

            try:
                # Try updating by native stored type
                res = await db.parsed_resumes.update_one({"_id": _id}, {"$set": update_doc})
                if res.matched_count == 0:
                    # try string id fallback if stored as string
                    res2 = await db.parsed_resumes.update_one({"_id": str(_id)}, {"$set": update_doc})
                    if res2.matched_count == 0:
                        print(f"⚠️  Could not match document _id={_id} to update.")
                    else:
                        updated += 1
                else:
                    updated += 1
            except Exception as e:
                print(f"❌ Update failed for _id={_id}: {e}")

        print(f"Processed {min(attempted, total)}/{total}...")

    print(f"✅ Done. Updated embeddings for {updated} resume(s). Attempts={attempted}. Dry-run={dry_run}.")


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill ANN index blobs + embeddings for resumes.")
    parser.add_argument("--owner", type=str, default=None, help="Owner userId to restrict the backfill.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of resumes to process.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMB_BATCH, help="Embedding batch size.")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if index_embedding exists.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to DB.")
    return parser.parse_args()


async def _amain() -> None:
    args = parse_args()
    await backfill_embeddings(
        owner=args.owner,
        limit=args.limit,
        batch_size=args.batch_size,
        recompute=args.recompute,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(_amain())
