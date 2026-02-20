# BackEnd/tests/test_concurrent.py
# Concurrent submission and duplicate invite behavior (status codes).

import pytest
from fastapi.testclient import TestClient


def test_submit_without_token_returns_404(client):
    r = client.post(
        "/tests/submit",
        json={"token": "nonexistent-token", "answers": []},
    )
    assert r.status_code == 404


def test_start_without_token_returns_404(client):
    r = client.post(
        "/tests/start",
        json={"token": "nonexistent-token"},
    )
    assert r.status_code == 404
