# BackEnd/tests/test_ownership.py
# Invite, save-selection, rerun, PDF: ownership enforcement.

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from main import app  # noqa: E402
from app.routers.auth_router import get_current_user


def _override_user(user_id: str, email: str = "test@test.local"):
    def _dep():
        return SimpleNamespace(id=user_id, email=email)
    app.dependency_overrides[get_current_user] = _dep


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_invite_returns_403_for_non_owner(client):
    _override_user("recruiter-A")
    r = client.post(
        "/tests/invite",
        headers={"Authorization": "Bearer fake", "Content-Type": "application/json"},
        json={
            "candidate_id": "candidate-owned-by-B",
            "question_count": 4,
            "test_type": "smart",
        },
    )
    assert r.status_code == 403
    assert "not authorized" in (r.json().get("detail") or "").lower()


def test_invite_returns_404_when_candidate_not_found(client):
    _override_user("recruiter-A")
    r = client.post(
        "/tests/invite",
        headers={"Authorization": "Bearer fake", "Content-Type": "application/json"},
        json={
            "candidate_id": "nonexistent-id-12345",
            "question_count": 4,
            "test_type": "smart",
        },
    )
    assert r.status_code in (403, 404)


def test_save_selection_requires_auth(client):
    r = client.post(
        "/history/save-selection",
        json={"selectedIds": ["id1"], "prompt": "test"},
    )
    assert r.status_code == 401


def test_save_selection_validates_body(client):
    _override_user("user1")
    r = client.post(
        "/history/save-selection",
        headers={"Authorization": "Bearer x", "Content-Type": "application/json"},
        json={"selectedIds": [], "prompt": "x"},
    )
    assert r.status_code == 400


def test_rerun_requires_auth(client):
    r = client.post(
        "/history/rerun/507f1f77bcf86cd799439011",
        json={"prompt": "refined"},
    )
    assert r.status_code == 401


def test_pdf_report_requires_auth(client):
    r = client.get("/tests/history/cand1/attempt1/report.pdf")
    assert r.status_code == 401


def test_pdf_report_returns_404_for_unknown_candidate(client: TestClient):
    _override_user("owner1")
    r = client.get(
        "/tests/history/nonexistent-candidate/attempt1/report.pdf",
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 404
