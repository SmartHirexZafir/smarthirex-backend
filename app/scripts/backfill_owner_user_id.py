# scripts/backfill_owner_user_id.py
"""
Backfill `ownerUserId` for existing CV (or any resource) documents.

Usage examples:
  # Dry-run on 'cvs' collection, just show what would change
  python scripts/backfill_owner_user_id.py --collection cvs --dry-run

  # Actually write changes
  python scripts/backfill_owner_user_id.py --collection cvs

  # Custom inference field priority & batch size
  python scripts/backfill_owner_user_id.py --collection cvs \
    --infer-fields userId,createdBy,owner,uploaderId,candidateUserId \
    --batch-size 500

Env fallbacks if app utils are unavailable:
  MONGO_URL=mongodb://localhost:27017  MONGO_DB=mydb  python scripts/backfill_owner_user_id.py --collection cvs
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

# --- Try to use your app's db helper first -------------------------------------
_db_mode = "unknown"  # 'motor' | 'pymongo' | 'env'
get_db = None  # type: ignore

try:
    # Your project path shows app/utils/mongo.py exists:contentReference[oaicite:1]{index=1}
    from app.utils.mongo import get_db as _app_get_db  # type: ignore
    get_db = _app_get_db
    _db_mode = "app"
except Exception:
    # NEW: if get_db is not exported by app.utils.mongo, gracefully fall back to its `db` object
    try:
        from app.utils.mongo import db as _app_db  # type: ignore

        def _fallback_get_db():
            # return the already-initialized database instance from the app
            return _app_db

        get_db = _fallback_get_db  # type: ignore
        _db_mode = "app"
    except Exception:
        # Fallback to env-based connection later
        _db_mode = "env"

# --- Utility: ObjectId optional import -----------------------------------------
try:
    from bson import ObjectId  # type: ignore
except Exception:
    ObjectId = None  # type: ignore

# NEW: Optional Motor type import for robust runtime detection
try:
    from motor.motor_asyncio import AsyncIOMotorDatabase  # type: ignore
except Exception:
    AsyncIOMotorDatabase = None  # type: ignore


def _is_awaitable(obj: Any) -> bool:
    return hasattr(obj, "__await__")


def _stringify(val: Any) -> Optional[str]:
    """Convert various id types to string."""
    if val is None:
        return None
    try:
        # Handle ObjectId -> str
        if ObjectId and isinstance(val, ObjectId):
            return str(val)
    except Exception:
        pass
    # NEW: lists/tuples — pick the first non-empty element
    if isinstance(val, (list, tuple)):
        for v in val:
            s = _stringify(v)
            if s:
                return s
        return None
    # dict with id/_id
    if isinstance(val, dict):
        for k in ("id", "_id", "userId", "ownerUserId"):
            if k in val and val[k]:
                return _stringify(val[k])
        return None
    # simple primitives
    if isinstance(val, (str, int)):
        return str(val)
    return None


def _infer_owner(doc: dict, fields_priority: Sequence[str]) -> Optional[str]:
    for f in fields_priority:
        if f in doc and doc[f] not in (None, ""):
            sid = _stringify(doc[f])
            if sid:
                return sid
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill ownerUserId for documents missing it."
    )
    p.add_argument(
        "--collection",
        required=True,
        help="Target Mongo collection name (e.g., 'cvs', 'candidates').",
    )
    p.add_argument(
        "--owner-field",
        default="ownerUserId",
        help="Owner field name to backfill (default: ownerUserId).",
    )
    p.add_argument(
        "--infer-fields",
        default="userId,createdBy,owner,uploaderId,candidateUserId,authorId",
        help=(
            "Comma-separated candidate fields (by priority) to infer owner from. "
            "Default: userId,createdBy,owner,uploaderId,candidateUserId,authorId"
        ),
    )
    p.add_argument(
        "--filter",
        default="{}",
        help="Additional JSON-like filter (simple Python dict literal) applied along with 'owner-field' is missing.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Bulk write batch size (default: 1000).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes, just print what would happen.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N docs (0 = no limit).",
    )
    return p.parse_args()


# ------------------------- Async (Motor) implementation -------------------------

async def _run_with_motor(
    db: Any,
    collection_name: str,
    owner_field: str,
    infer_fields: Sequence[str],
    extra_filter: dict,
    batch_size: int,
    dry_run: bool,
    limit: int,
) -> Tuple[int, int, int]:
    """
    Uses Motor's async API. Keeps original behavior but switches to collection.bulk_write
    for reliability (instead of raw db.command('bulkWrite')).
    """
    coll = db[collection_name]

    # Include both "missing" and "null/empty" owner_field cases (additive)
    query = {"$or": [{owner_field: {"$exists": False}}, {owner_field: {"$in": [None, ""]}}]}
    if extra_filter:
        query.update(extra_filter)

    # Projection keeps payload smaller
    cursor = coll.find(query, projection={owner_field: 1, **{f: 1 for f in infer_fields}})
    processed = updated = skipped = 0
    ops = []

    # Import here to avoid mandatory dependency at import time
    try:
        from pymongo import UpdateOne  # type: ignore
    except Exception:
        UpdateOne = None  # type: ignore

    async for doc in cursor:
        if limit and processed >= limit:
            break
        processed += 1

        if owner_field in doc and doc[owner_field]:
            continue

        inferred = _infer_owner(doc, infer_fields)
        if not inferred:
            skipped += 1
            continue

        if dry_run or UpdateOne is None:
            print(f"[DRY] would set {owner_field}={inferred} for _id={doc.get('_id')}")
        else:
            ops.append(
                UpdateOne({"_id": doc["_id"]}, {"$set": {owner_field: inferred}})
            )
            if len(ops) >= batch_size:
                res = await coll.bulk_write(ops, ordered=False)
                updated += res.modified_count
                ops = []

    if ops and not dry_run and UpdateOne is not None:
        res = await coll.bulk_write(ops, ordered=False)
        updated += res.modified_count

    return processed, updated, skipped


# ------------------------ Sync (PyMongo) implementation -------------------------

def _run_with_pymongo(
    db: Any,
    collection_name: str,
    owner_field: str,
    infer_fields: Sequence[str],
    extra_filter: dict,
    batch_size: int,
    dry_run: bool,
    limit: int,
) -> Tuple[int, int, int]:
    from pymongo import UpdateOne  # type: ignore

    coll = db[collection_name]

    # Include both "missing" and "null/empty" owner_field cases (additive)
    query = {"$or": [{owner_field: {"$exists": False}}, {owner_field: {"$in": [None, ""]}}]}
    if extra_filter:
        query.update(extra_filter)

    processed = updated = skipped = 0
    ops: List[UpdateOne] = []

    cursor = coll.find(query, projection={owner_field: 1, **{f: 1 for f in infer_fields}})
    for doc in cursor:
        if limit and processed >= limit:
            break
        processed += 1

        if owner_field in doc and doc[owner_field]:
            continue

        inferred = _infer_owner(doc, infer_fields)
        if not inferred:
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY] would set {owner_field}={inferred} for _id={doc.get('_id')}")
        else:
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {owner_field: inferred}}))
            if len(ops) >= batch_size:
                res = coll.bulk_write(ops, ordered=False)
                updated += res.modified_count
                ops = []

    if ops and not dry_run:
        res = coll.bulk_write(ops, ordered=False)
        updated += res.modified_count

    return processed, updated, skipped


# -------------------------------- Entry point -----------------------------------

def _env_connect():
    """Fallback PyMongo connection via env MONGO_URL + MONGO_DB."""
    from pymongo import MongoClient  # type: ignore

    mongo_url = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
    mongo_db = os.getenv("MONGO_DB") or os.getenv("MONGODB_DB") or "test"
    client = MongoClient(mongo_url)
    return client[mongo_db]


def main():
    args = _parse_args()
    infer_fields = [f.strip() for f in args.infer_fields.split(",") if f.strip()]
    try:
        extra_filter = eval(args.filter, {"__builtins__": {}}, {})  # simple literal-only eval
        if not isinstance(extra_filter, dict):
            raise ValueError
    except Exception:
        print("--filter must be a simple dict literal, e.g. \"{'status':'parsed'}\"")
        sys.exit(2)

    if _db_mode == "app" and get_db:
        # Try using app's helper, which could be sync or async (Motor).
        db_candidate = get_db()
        if _is_awaitable(db_candidate):
            async def _run_async():
                db = await db_candidate
                processed, updated, skipped = await _run_with_motor(
                    db=db,
                    collection_name=args.collection,
                    owner_field=args.owner_field,
                    infer_fields=infer_fields,
                    extra_filter=extra_filter,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                    limit=args.limit,
                )
                print(f"\nDone (MOTOR/awaitable). Processed: {processed}, Updated: {updated}, Skipped: {skipped}")
            asyncio.run(_run_async())
        else:
            # Detect Motor vs PyMongo instance at runtime
            db = db_candidate
            if AsyncIOMotorDatabase is not None and isinstance(db, AsyncIOMotorDatabase):
                async def _run_async_motor():
                    processed, updated, skipped = await _run_with_motor(
                        db=db,
                        collection_name=args.collection,
                        owner_field=args.owner_field,
                        infer_fields=infer_fields,
                        extra_filter=extra_filter,
                        batch_size=args.batch_size,
                        dry_run=args.dry_run,
                        limit=args.limit,
                    )
                    print(f"\nDone (MOTOR). Processed: {processed}, Updated: {updated}, Skipped: {skipped}")
                asyncio.run(_run_async_motor())
            else:
                # Assume PyMongo db
                processed, updated, skipped = _run_with_pymongo(
                    db=db,
                    collection_name=args.collection,
                    owner_field=args.owner_field,
                    infer_fields=infer_fields,
                    extra_filter=extra_filter,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                    limit=args.limit,
                )
                print(f"\nDone (PYMONGO via app). Processed: {processed}, Updated: {updated}, Skipped: {skipped}")
    else:
        # Env fallback
        db = _env_connect()
        processed, updated, skipped = _run_with_pymongo(
            db=db,
            collection_name=args.collection,
            owner_field=args.owner_field,
            infer_fields=infer_fields,
            extra_filter=extra_filter,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        print(f"\nDone (PYMONGO via env). Processed: {processed}, Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
