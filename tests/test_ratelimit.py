from synthwerk_mascot.ratelimit import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limit_client_over_limit_429_with_retry_after() -> None:
    clock = Clock()
    limiter = RateLimiter(per_client=2, global_limit=100, window_s=60, clock=clock)
    assert limiter.acquire("10.0.0.1") is None
    clock.now += 10
    assert limiter.acquire("10.0.0.1") is None
    clock.now += 5
    assert limiter.acquire("10.0.0.1") == 45  # the first request leaves the window at +60


def test_rate_limit_window_slides_client_may_ask_again() -> None:
    clock = Clock()
    limiter = RateLimiter(per_client=1, global_limit=100, window_s=60, clock=clock)
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is not None
    clock.now += 60
    assert limiter.acquire("a") is None


def test_rate_limit_other_client_not_affected() -> None:
    limiter = RateLimiter(per_client=1, global_limit=100, window_s=60, clock=Clock())
    assert limiter.acquire("a") is None
    assert limiter.acquire("b") is None


def test_rate_limit_global_limit_stops_every_client() -> None:
    limiter = RateLimiter(per_client=5, global_limit=2, window_s=60, clock=Clock())
    assert limiter.acquire("a") is None
    assert limiter.acquire("b") is None
    assert limiter.acquire("c") == 60


def test_rate_limit_refused_request_not_counted() -> None:
    clock = Clock()
    limiter = RateLimiter(per_client=1, global_limit=100, window_s=60, clock=clock)
    limiter.acquire("a")
    for _ in range(5):
        limiter.acquire("a")
    clock.now += 60
    assert limiter.acquire("a") is None


def test_rate_limit_table_full_expired_clients_pruned() -> None:
    clock = Clock()
    limiter = RateLimiter(per_client=1, global_limit=100, window_s=60, max_clients=2, clock=clock)
    limiter.acquire("a")
    limiter.acquire("b")
    assert limiter.acquire("c") == 60  # full, nobody expired
    clock.now += 61
    assert limiter.acquire("c") is None
    assert limiter.tracked_clients == 1


def test_rate_limit_key_is_keyed_hash_not_the_address() -> None:
    first = RateLimiter(per_client=1, global_limit=1, window_s=1)
    second = RateLimiter(per_client=1, global_limit=1, window_s=1)
    assert b"10.0.0.1" not in first.key("10.0.0.1")
    assert first.key("10.0.0.1") == first.key("10.0.0.1")
    assert first.key("10.0.0.1") != second.key("10.0.0.1")
