# BackEnd/tests/test_auth.py

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_dashboard_requires_auth(client):
    r = client.get("/dashboard/summary")
    assert r.status_code == 401


def test_dashboard_with_invalid_token(client):
    r = client.get(
        "/dashboard/summary",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert r.status_code == 401


def test_logout_returns_ok(client):
    r = client.post("/logout")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_auth_logout_returns_ok(client):
    r = client.post("/auth/logout")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_login_missing_body(client):
    r = client.post("/auth/login", json={})
    assert r.status_code == 422
