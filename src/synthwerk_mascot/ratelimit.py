"""Strict in-memory rate limit: a sliding window per client and one for all clients.

The client key is an HMAC of the client address with a random per-process key, so the
limiter never holds a plain IP address, and a restart forgets every key. Nothing is written
to disk. One process holds the state: the mascot runs one worker.
"""

import hashlib
import hmac
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        per_client: int,
        global_limit: int,
        window_s: float,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_client = per_client
        self._global_limit = global_limit
        self._window = window_s
        self._max_clients = max_clients
        self._clock = clock
        self._salt = secrets.token_bytes(32)
        self._clients: dict[bytes, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def key(self, client: str) -> bytes:
        return hmac.new(self._salt, client.encode(), hashlib.sha256).digest()[:16]

    @property
    def tracked_clients(self) -> int:
        return len(self._clients)

    def _expire(self, stamps: deque[float], now: float) -> None:
        while stamps and stamps[0] <= now - self._window:
            stamps.popleft()

    def _retry_after(self, stamps: deque[float], now: float) -> int:
        return max(1, math.ceil(stamps[0] + self._window - now))

    def _prune(self, now: float) -> None:
        for key in list(self._clients):
            stamps = self._clients[key]
            self._expire(stamps, now)
            if not stamps:
                del self._clients[key]

    def acquire(self, client: str) -> int | None:
        """None when the request may pass (and counts it), else the seconds to wait."""
        key = self.key(client)
        with self._lock:
            now = self._clock()
            self._expire(self._all, now)
            if len(self._all) >= self._global_limit:
                return self._retry_after(self._all, now)
            stamps = self._clients.get(key)
            if stamps is None:
                if len(self._clients) >= self._max_clients:
                    self._prune(now)
                if len(self._clients) >= self._max_clients:
                    return math.ceil(self._window)
                stamps = self._clients[key] = deque()
            self._expire(stamps, now)
            if len(stamps) >= self._per_client:
                return self._retry_after(stamps, now)
            stamps.append(now)
            self._all.append(now)
            return None
