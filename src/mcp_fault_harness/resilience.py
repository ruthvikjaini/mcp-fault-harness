"""Client-side resilience mechanisms: circuit breaker, idempotency, cache.

These are the three mechanisms that turn a fault from a failed task
into a recovered one — or, when misapplied, into a new failure. Each
is stateful, deterministic, and driven by the same virtual clock as
the rest of the harness, so experiments stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-tool breaker: fail fast instead of hammering a dead tool.

    Opens after ``failure_threshold`` consecutive failures, stays open
    for ``reset_timeout_ms`` of virtual time, then allows one probe
    call (half-open). A successful probe closes the breaker; a failed
    probe re-opens it. Successes reset the consecutive-failure count.
    """

    failure_threshold: int = 3
    reset_timeout_ms: int = 1_000
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at_ms: int = 0
    probes: int = 0
    short_circuits: int = 0

    def allow(self, now_ms: int) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if now_ms - self.opened_at_ms >= self.reset_timeout_ms:
                self.state = BreakerState.HALF_OPEN
                self.probes += 1
                return True
            self.short_circuits += 1
            return False
        # HALF_OPEN: only the probe call is allowed; the caller makes
        # exactly one call and reports the outcome.
        return True

    def on_success(self) -> None:
        self.consecutive_failures = 0
        self.state = BreakerState.CLOSED

    def on_failure(self, now_ms: int) -> None:
        self.consecutive_failures += 1
        if self.state == BreakerState.HALF_OPEN:
            self.state = BreakerState.OPEN
            self.opened_at_ms = now_ms
            return
        if self.consecutive_failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self.opened_at_ms = now_ms


@dataclass
class IdempotencyStore:
    """Client-side record of write results keyed by idempotency key.

    The store is only half the mechanism: the server must also honour
    the key (see :class:`servers.WriteServer`). The client store lets
    the agent short-circuit a retry without calling the server again
    when it already has the result; the server-side store is what
    prevents a duplicate effect when the client retries after an
    ambiguous timeout.
    """

    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.results.get(key)

    def put(self, key: str, result: dict[str, Any]) -> None:
        self.results[key] = dict(result)


@dataclass
class FallbackCache:
    """Stale-but-usable cached values, explicitly marked as such.

    Entries are keyed by ``tool`` plus the sorted call arguments, so a
    cached price for one item is never served for another. A fallback
    value is never presented as live evidence: lookups return the
    cached payload with ``source="cache"`` and ``stale=True``, so
    scoring can give partial credit instead of full credit, and so an
    agent that checks freshness never mistakes it for fresh data.
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def key(tool: str, arguments: dict[str, Any]) -> str:
        parts = ",".join(f"{k}={v}" for k, v in sorted(arguments.items()))
        return f"{tool}:{parts}"

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self.entries[key] = dict(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        payload = dict(entry)
        payload["stale"] = True
        return payload
