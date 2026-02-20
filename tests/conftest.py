# BackEnd/tests/conftest.py
# Pytest fixtures. Run: pytest tests/ -v
# For integration tests set TEST_USER_EMAIL, TEST_USER_PASSWORD and use real DB.

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET", os.getenv("JWT_SECRET", "test-secret-not-for-prod"))

from main import app  # noqa: E402
from app.routers.auth_router import get_current_user  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


def _make_user(user_id: str, email: str = "user@test.local"):
    return SimpleNamespace(id=user_id, email=email)


@pytest.fixture
def user_a():
    return _make_user("user-a-id", "a@test.local")


@pytest.fixture
def user_b():
    return _make_user("user-b-id", "b@test.local")


@pytest.fixture
def auth_headers_integration(client):
    """Real login; requires TEST_USER_EMAIL, TEST_USER_PASSWORD and verified user in DB."""
    email = os.getenv("TEST_USER_EMAIL")
    password = os.getenv("TEST_USER_PASSWORD")
    if not email or not password:
        return None
    r = client.post("/auth/login", json={"email": email, "password": password})
    if r.status_code != 200:
        return None
    token = r.json().get("token")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
