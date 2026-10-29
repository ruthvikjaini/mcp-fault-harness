from mcp_fault_harness.resilience import (
    BreakerState,
    CircuitBreaker,
    FallbackCache,
    IdempotencyStore,
)


def test_breaker_opens_after_threshold_and_recovers_on_probe():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_ms=100)
    assert breaker.allow(0)
    breaker.on_failure(0)
    breaker.on_failure(10)
    assert breaker.state == BreakerState.OPEN
    assert not breaker.allow(50)
    assert breaker.short_circuits == 1
    assert breaker.allow(200)  # half-open probe
    breaker.on_success()
    assert breaker.state == BreakerState.CLOSED
    assert breaker.consecutive_failures == 0


def test_breaker_failed_probe_reopens():
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_ms=100)
    breaker.on_failure(0)
    assert breaker.state == BreakerState.OPEN
    assert breaker.allow(150)
    breaker.on_failure(150)
    assert breaker.state == BreakerState.OPEN


def test_idempotency_store_roundtrip():
    store = IdempotencyStore()
    assert store.get("k") is None
    store.put("k", {"effect_id": "e1"})
    assert store.get("k") == {"effect_id": "e1"}


def test_cache_key_separates_arguments():
    cache = FallbackCache()
    cache.put(FallbackCache.key("catalog_price", {"item_id": "widget-a"}), {"value": 120.0, "as_of": 0})
    assert cache.get(FallbackCache.key("catalog_price", {"item_id": "widget-b"})) is None
    entry = cache.get(FallbackCache.key("catalog_price", {"item_id": "widget-a"}))
    assert entry["value"] == 120.0 and entry["stale"] is True
