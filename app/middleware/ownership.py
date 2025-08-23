# app/middleware/ownership.py
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Union, Awaitable, Dict
from fastapi import Depends, HTTPException, Path
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND
from bson import ObjectId

# Try to import your existing auth dependency to get the current user.
# Adjust the import path/name if your project exposes it elsewhere.
try:
    # expected to return an object/dict with an "id" or "_id" attribute/field
    from app.routers.auth_router import get_current_user  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Ownership middleware couldn't import get_current_user. "
        "Update the import in ownership.py to match your auth dependency."
    ) from exc


# --- Type helpers ----------------------------------------------------------------

JSONDoc = Dict[str, Any]

class SupportsFindOne(Protocol):
    """Subset of Motor/PyMongo Collection used here."""
    def find_one(self, *args: Any, **kwargs: Any) -> Union[JSONDoc, None]: ...
    # For Motor async collections:
    async def find_one(self, *args: Any, **kwargs: Any) -> Union[JSONDoc, None]: ...  # type: ignore[override]


def _to_object_id(id_value: Union[str, ObjectId], id_field: str = "_id") -> Union[str, ObjectId]:
    """
    Convert a string id to ObjectId if the target id field looks like Mongo _id.
    Otherwise return as-is (useful if you store ids as strings).
    """
    if id_field == "_id" and isinstance(id_value, str):
        try:
            return ObjectId(id_value)
        except Exception:
            # If it isn't a valid ObjectId, let it pass through; lookup will 404 later
            return id_value
    return id_value


def _extract_user_id(user: Any) -> Optional[str]:
    """
    Accepts either a dict-like or an object and returns the string user id.
    Looks for 'id' then '_id'.
    """
    if user is None:
        return None
    # dict-like
    if isinstance(user, dict):
        return str(user.get("id") or user.get("_id") or "")
    # object with attrs
    return str(getattr(user, "id", None) or getattr(user, "_id", "") or "")


# --- Core ownership checks --------------------------------------------------------

def assert_owner(
    doc: Optional[JSONDoc],
    user_id: str,
    *,
    owner_field: str = "ownerUserId",
) -> JSONDoc:
    """
    Ensure that `doc` exists and belongs to `user_id`.
    Returns the doc if OK; raises 404/403 otherwise.
    """
    if not doc:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Resource not found")

    doc_owner = str(doc.get(owner_field, ""))
    if not doc_owner:
        # Treat missing owner as not found to avoid leaking existence
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Resource not found")

    if doc_owner != str(user_id):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Forbidden")

    return doc


async def fetch_and_assert_owner(
    collection: SupportsFindOne,
    resource_id: Union[str, ObjectId],
    user_id: str,
    *,
    id_field: str = "_id",
    owner_field: str = "ownerUserId",
) -> JSONDoc:
    """
    Load a document by id from `collection` and ensure it belongs to `user_id`.
    Works with both Motor (async) and PyMongo (sync) collections.
    """
    query_id = _to_object_id(resource_id, id_field=id_field)
    query = {id_field: query_id}

    # Motor returns awaitable; PyMongo returns dict
    maybe_awaitable = collection.find_one(query)  # type: ignore[call-arg]
    if hasattr(maybe_awaitable, "__await__"):
        doc = await maybe_awaitable  # Motor
    else:
        doc = maybe_awaitable  # type: ignore[assignment]

    return assert_owner(doc, user_id, owner_field=owner_field)


# --- FastAPI dependency factory (optional) ---------------------------------------

def OwnerGuard(
    *,
    get_collection: Callable[[], SupportsFindOne],
    param_name: str = "id",
    id_field: str = "_id",
    owner_field: str = "ownerUserId",
):
    """
    Create a FastAPI dependency that:
      1) reads a path param (default 'id'),
      2) reads current_user via your existing get_current_user,
      3) fetches the doc from `get_collection()`,
      4) asserts ownership or raises 404/403,
      5) returns the document.

    Usage:

        # In your router
        from fastapi import APIRouter, Depends
        from app.middleware.ownership import OwnerGuard
        from app.utils.mongo import get_db  # your own helper

        router = APIRouter(prefix="/cvs", tags=["CVs"])

        def cv_collection():
            db = get_db()
            return db["cvs"]

        @router.get("/{id}")
        async def get_cv(
            doc = Depends(OwnerGuard(get_collection=cv_collection, param_name="id"))
        ):
            return doc

    You can also change id/owner field names if your schema differs.
    """
    async def _dependency(
        resource_id: str = Path(..., alias=param_name),
        current_user: Any = Depends(get_current_user),
    ) -> JSONDoc:
        user_id = _extract_user_id(current_user)
        if not user_id:
            # get_current_user should normally 401 before we get here,
            # but just in case:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Forbidden")

        collection = get_collection()
        return await fetch_and_assert_owner(
            collection=collection,
            resource_id=resource_id,
            user_id=user_id,
            id_field=id_field,
            owner_field=owner_field,
        )

    return _dependency


# --- Convenience filter for list/search queries ----------------------------------

def user_scope_filter(
    user_id: str,
    *,
    owner_field: str = "ownerUserId",
    base_filter: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Quick helper to build a Mongo filter that always scopes to the user.

    Example:
        q = user_scope_filter(current_user.id, base_filter={"status": "parsed"})
        docs = await coll.find(q).to_list(length=100)
    """
    f = dict(base_filter or {})
    f[owner_field] = str(user_id)
    return f
