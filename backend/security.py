"""API-key auth and per-client rate limiting for /api/ask.

Every question spends LLM quota (and sometimes Open-Meteo quota), so the endpoint
is gated. Both checks live here so main.py stays a thin route."""

import hmac
import os
import threading
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse

MAX_TRACKED_CLIENTS = 10_000  # bounds memory if someone sprays requests from many addresses


class RateLimiter:
    """Sliding-window limiter: at most `per_minute` requests in any 60 s and
    `per_day` in any 24 h, per client. In-memory, so per-process -- fine for one
    uvicorn worker; use Redis (or a gateway) if you run several."""

    def __init__(self, per_minute: int, per_day: int, clock=time.monotonic):
        self.per_minute, self.per_day, self._clock = per_minute, per_day, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> int:
        """Records a request. Returns 0 if allowed, else seconds until it would be."""
        now = self._clock()
        with self._lock:
            if client not in self._hits and len(self._hits) >= MAX_TRACKED_CLIENTS:
                self._evict_idle(now)
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= 86400:
                hits.popleft()
            recent = [t for t in hits if now - t < 60]
            if len(recent) >= self.per_minute:
                return max(1, int(60 - (now - recent[0])) + 1)
            if len(hits) >= self.per_day:
                return max(1, int(86400 - (now - hits[0])) + 1)
            hits.append(now)
            return 0

    def _evict_idle(self, now: float) -> None:
        for client in [c for c, h in self._hits.items() if not h or now - h[-1] >= 86400]:
            del self._hits[client]
        if len(self._hits) >= MAX_TRACKED_CLIENTS:  # still full: drop the oldest-seen client
            del self._hits[min(self._hits, key=lambda c: self._hits[c][-1])]


limiter = RateLimiter(
    per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "10")),
    per_day=int(os.environ.get("RATE_LIMIT_PER_DAY", "200")),
)


def auth_enabled() -> bool:
    return bool(os.environ.get("APP_API_KEY"))


def guard(request: Request) -> JSONResponse | None:
    """Returns an error response to send, or None if the request may proceed.
    Rate limiting runs first and counts failed-auth attempts too, so the key
    can't be brute-forced at speed. Clients are identified by socket address;
    behind a reverse proxy, configure uvicorn's --proxy-headers so that is the
    real client rather than the proxy."""
    client = request.client.host if request.client else "unknown"
    retry_after = limiter.check(client)
    if retry_after:
        return JSONResponse(
            {"error": f"rate limit exceeded; retry in {retry_after}s"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    expected = os.environ.get("APP_API_KEY")
    if expected:
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):  # constant-time compare
            return JSONResponse({"error": "invalid or missing API key"}, status_code=401)
    return None
