# BackEnd/tests/test_pdf_route.py
# PDF report route: success shape and unauthorized.

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace
from main import app  # noqa: E402
from app.routers.auth_router import get_current_user


def _override_user(user_id: str):
    def _dep():
        return SimpleNamespace(id=user_id, email="u@t.local")
    app.dependency_overrides[get_current_user] = _dep


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_pdf_report_unauthorized_without_token(client: TestClient):
    r = client.get("/tests/history/any/any/report.pdf")
    assert r.status_code == 401


def test_pdf_report_not_found_candidate_returns_404(client):
    _override_user("owner-1")
    r = client.get(
        "/tests/history/fake-candidate-id/fake-attempt-id/report.pdf",
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 404
    assert "detail" in r.json()
