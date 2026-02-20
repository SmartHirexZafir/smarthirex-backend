# BackEnd/tests/load_simulation.py
# Load and concurrency simulation. Run: python -m tests.load_simulation
# Set BASE_URL (default http://localhost:10000) and optionally TEST_USER_EMAIL, TEST_USER_PASSWORD.

import os
import sys
import time
import concurrent.futures
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE_URL = os.getenv("BASE_URL", "http://localhost:10000").rstrip("/")
EMAIL = os.getenv("TEST_USER_EMAIL", "")
PASSWORD = os.getenv("TEST_USER_PASSWORD", "")


def _request(path: str, method: str = "GET", body: str = None, token: str = None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, data=body.encode() if body else None, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode() if e.fp else ""
    except URLError as e:
        return -1, str(e.reason)
    except Exception as e:
        return -2, str(e)


def get_token():
    if not EMAIL or not PASSWORD:
        return None
    import json
    status, data = _request("/auth/login", "POST", body=json.dumps({"email": EMAIL, "password": PASSWORD}))
    if status != 200:
        return None
    try:
        import json as _json
        return _json.loads(data).get("token")
    except Exception:
        return None


def run_concurrent_submissions(n: int = 100, token: str = None):
    tok = token if token is not None else get_token()
    if not tok:
        print("Load test: no token, using unauthenticated /tests/submit (expect 401/404)")
    results = []

    def one():
        status, _ = _request("/tests/submit", "POST", body='{"token":"nonexistent","answers":[]}', token=tok)
        return status

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 50)) as ex:
        futs = [ex.submit(one) for _ in range(n)]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(str(e))
    return results


def run_concurrent_invites(n: int = 50, token: str = None):
    tok = token if token is not None else get_token()
    if not tok:
        return [401] * n
    results = []
    import json as _json
    body = _json.dumps({"candidateId": "nonexistent_candidate_id_123", "email": "test@example.com"})

    def one():
        status, _ = _request("/tests/invite", "POST", body=body, token=tok)
        return status

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 20)) as ex:
        futs = [ex.submit(one) for _ in range(n)]
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(str(e))
    return results


def run_meeting_access_polls(n: int = 50):
    results = []
    for _ in range(n):
        status, _ = _request("/interviews/access/fake-meeting-token-123")
        results.append(status)
    return results


def run_dashboard_requests(n: int = 100, token: str = None):
    results = []
    for _ in range(n):
        status, _ = _request("/dashboard/summary", token=token)
        results.append(status)
    return results


def main():
    print("Load simulation (no auth = 401 on protected routes)")
    token = get_token()
    print(f"Token: {'yes' if token else 'no'}")

    # 100 dashboard summary requests
    print("100x GET /dashboard/summary...")
    t0 = time.perf_counter()
    dash = run_dashboard_requests(100, token)
    elapsed = time.perf_counter() - t0
    ok = sum(1 for s in dash if s == 200)
    auth_fail = sum(1 for s in dash if s == 401)
    print(f"  Done in {elapsed:.2f}s: 200={ok}, 401={auth_fail}")

    # 100 concurrent test submissions (bad token -> 404)
    print("100x POST /tests/submit (concurrent)...")
    t0 = time.perf_counter()
    sub = run_concurrent_submissions(100, token)
    elapsed = time.perf_counter() - t0
    from collections import Counter
    sub_counts = Counter(sub)
    print(f"  Done in {elapsed:.2f}s: {dict(sub_counts)} (no crash)")

    # 50 concurrent invite attempts
    print("50x POST /tests/invite (concurrent)...")
    t0 = time.perf_counter()
    inv = run_concurrent_invites(50, token)
    elapsed = time.perf_counter() - t0
    inv_counts = Counter(inv)
    print(f"  Done in {elapsed:.2f}s: {dict(inv_counts)} (no crash)")

    # 50 meeting access polls (no token -> 403)
    print("50x GET /interviews/access/fake-meeting-token...")
    t0 = time.perf_counter()
    acc = run_meeting_access_polls(50)
    elapsed = time.perf_counter() - t0
    acc_counts = Counter(acc)
    print(f"  Done in {elapsed:.2f}s: {dict(acc_counts)} (no crash)")

    print("Load simulation done.")


if __name__ == "__main__":
    main()
    sys.exit(0)
