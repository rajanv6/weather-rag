import json

import pytest
from fastapi.testclient import TestClient

import main
import security
from security import RateLimiter


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- RateLimiter ---------------------------------------------------------------------

def test_limiter_blocks_after_the_per_minute_budget_and_recovers():
    clock = Clock()
    limiter = RateLimiter(per_minute=3, per_day=100, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [0, 0, 0]
    wait = limiter.check("a")
    assert 1 <= wait <= 61
    clock.now += 61
    assert limiter.check("a") == 0


def test_limiter_enforces_the_daily_budget():
    clock = Clock()
    limiter = RateLimiter(per_minute=100, per_day=5, clock=clock)
    for _ in range(5):
        assert limiter.check("a") == 0
        clock.now += 30  # stays under the per-minute limit
    assert limiter.check("a") > 60  # now it's the day window talking
    clock.now += 86400
    assert limiter.check("a") == 0


def test_limiter_tracks_clients_separately():
    limiter = RateLimiter(per_minute=1, per_day=10, clock=Clock())
    assert limiter.check("a") == 0
    assert limiter.check("a") > 0
    assert limiter.check("b") == 0


def test_limiter_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(security, "MAX_TRACKED_CLIENTS", 3)
    limiter = RateLimiter(per_minute=5, per_day=10, clock=Clock())
    for i in range(10):
        limiter.check(f"client-{i}")
    assert len(limiter._hits) <= 3


# --- /api/ask: auth + rate limit end to end (LLM and retrieval stubbed out) -------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.setattr(security, "limiter", RateLimiter(per_minute=100, per_day=1000))
    monkeypatch.setattr(main.rag, "retrieve", lambda q: [])
    monkeypatch.setattr(main, "answer_stream", lambda q, r: iter([{"type": "token", "text": "hi"}]))
    return TestClient(main.app)  # not used as a context manager, so the startup index build is skipped


def ask(client, key=None, question="weather in Chennai?"):
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post("/api/ask", json={"question": question}, headers=headers)


def events(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_open_when_no_key_is_configured(client):
    resp = ask(client)
    assert resp.status_code == 200
    assert [e["type"] for e in events(resp)] == ["retrieved", "token", "done"]


def test_missing_or_wrong_key_is_rejected(client, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    assert ask(client).status_code == 401
    wrong = ask(client, key="nope")
    assert wrong.status_code == 401 and "API key" in wrong.json()["error"]


def test_correct_key_is_accepted(client, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    assert ask(client, key="s3cret").status_code == 200


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(security, "limiter", RateLimiter(per_minute=2, per_day=100))
    assert [ask(client).status_code for _ in range(2)] == [200, 200]
    blocked = ask(client)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "rate limit" in blocked.json()["error"]


def test_failed_auth_attempts_count_toward_the_limit(client, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "s3cret")
    monkeypatch.setattr(security, "limiter", RateLimiter(per_minute=2, per_day=100))
    assert [ask(client, key="guess").status_code for _ in range(2)] == [401, 401]
    assert ask(client, key="s3cret").status_code == 429  # can't brute-force at speed


def test_empty_question_is_a_400_not_an_llm_call(client):
    resp = ask(client, question="   ")
    assert resp.status_code == 400 and resp.json()["error"] == "question is required"
