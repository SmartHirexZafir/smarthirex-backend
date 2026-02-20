# BackEnd/tests/test_meeting_access.py
# Meeting access token validation.

import pytest
from fastapi.testclient import TestClient


def test_meeting_access_without_token_returns_403(client):
    r = client.get("/interviews/access/fake-meeting-token")
    assert r.status_code == 403


def test_meeting_by_token_requires_auth(client):
    r = client.get("/interviews/by-token/fake-token")
    assert r.status_code == 401
