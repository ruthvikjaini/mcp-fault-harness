"""Seedable, protocol-aware fault injection.

The fault model has two axes, following the design described in
Iannillo et al. (2026): a failure *mode* (what goes wrong) and a
*persistence* (how long it lasts). A :class:`FaultSchedule` is created
per run from a seed, so the same seed and the same call sequence always
produce the same faults, while different policies — which make different
numbers of calls — see different fault streams, as they would against a
real flaky server.

Modes fall into two families:

* explicit failures (:class:`ToolError`): the caller sees the failure.
  ``timeout``, ``transport_error``, ``server_error``,
  ``rate_limited``, ``auth_expired``, ``malformed``.
* corrupt successes: a payload arrives but is wrong — ``stale`` (an
  old value), ``partial`` (a required field missing), ``empty`` (no
  value at all), ``slow`` (arrives, but late). These only hurt callers
  that do not validate, which is the silent-damage family.

``timeout_after_apply`` is the ambiguous case that makes retries
unsafe for writes: the effect happened, but the response was lost, so
a retry that is not idempotent applies the effect twice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mcp_fault_harness.protocol import ToolError, ToolResult


class FaultMode(str, Enum):
    TIMEOUT = "timeout"
    TIMEOUT_AFTER_APPLY = "timeout_after_apply"
    TRANSPORT_ERROR = "transport_error"
    SERVER_ERROR = "server_error"
    RATE_LIMITED = "rate_limited"
    AUTH_EXPIRED = "auth_expired"
    MALFORMED = "malformed"
    STALE = "stale"
    PARTIAL = "partial"
    EMPTY = "empty"
    SLOW = "slow"


class Persistence(str, Enum):
    TRANSIENT = "transient"  # faults exactly one call, then clears
    INTERMITTENT = "intermittent"  # each call faults with probability p
    PERMANENT = "permanent"  # every call to an affected tool faults


#: Modes that are worth retrying with the same arguments. Auth failures
#: are excluded: retrying expired credentials with the same credentials
#: cannot succeed. Corrupt successes are not errors, so they are not
#: retried here either — they are caught by validation instead.
RETRYABLE_MODES = frozenset(
    {
        FaultMode.TIMEOUT,
        FaultMode.TIMEOUT_AFTER_APPLY,
        FaultMode.TRANSPORT_ERROR,
        FaultMode.SERVER_ERROR,
        FaultMode.RATE_LIMITED,
        FaultMode.MALFORMED,
        FaultMode.SLOW,
    }
)

#: Which error kinds a retry policy will retry (protocol-level names).
RETRYABLE_ERROR_KINDS = frozenset(
    {
        "timeout",
        "timeout_after_apply",
        "transport_error",
        "server_error",
        "rate_limited",
        "malformed",
    }
)


@dataclass(frozen=True)
class FaultProfile:
    """The mix of modes and persistences for one experiment condition.

    ``mode_weights`` and ``persistence_weights`` are relative weights.
    ``fault_probability`` is the per-call chance that a call is faulted
    at all (intermittent tools use it per call; transient/permanent
    tools use it to decide whether the tool is affected). ``affected``
    tools are chosen per run from the seed.
    """

    name: str
    mode_weights: dict[str, float]
    persistence_weights: dict[str, float]
    fault_probability: float = 0.0
    intermittent_probability: float = 0.35

    def sample_mode(self, rng: random.Random) -> FaultMode:
        modes = list(self.mode_weights)
        weights = [self.mode_weights[m] for m in modes]
        return FaultMode(rng.choices(modes, weights=weights, k=1)[0])

    def sample_persistence(self, rng: random.Random) -> Persistence:
        names = list(self.persistence_weights)
        weights = [self.persistence_weights[n] for n in names]
        return Persistence(rng.choices(names, weights=weights, k=1)[0])


FAULT_FREE = FaultProfile(
    name="fault_free", mode_weights={}, persistence_weights={}, fault_probability=0.0
)

TRANSIENT_PROFILE = FaultProfile(
    name="transient",
    mode_weights={
        "timeout": 3.0,
        "transport_error": 2.0,
        "server_error": 2.0,
        "rate_limited": 1.0,
        "malformed": 1.0,
        "stale": 1.0,
    },
    persistence_weights={"transient": 1.0},
    fault_probability=0.30,
)

INTERMITTENT_PROFILE = FaultProfile(
    name="intermittent",
    mode_weights={
        "timeout": 3.0,
        "transport_error": 2.0,
        "server_error": 2.0,
        "rate_limited": 1.0,
        "malformed": 1.0,
        "stale": 1.0,
        "partial": 1.0,
        "empty": 1.0,
        "slow": 1.0,
    },
    persistence_weights={"intermittent": 1.0},
    fault_probability=0.35,
    intermittent_probability=0.35,
)

#: Permanent-heavy stress profile: most affected tools stay broken,
#: mirroring the permanent-heavy stress profile in Iannillo et al.
#: The weights are this project's assumption, stated in
#: ``docs/RESEARCH.md`` — the paper reports the profile is
#: permanent-heavy, not a mode-by-mode breakdown.
STRESS_PROFILE = FaultProfile(
    name="stress",
    mode_weights={
        "timeout": 2.0,
        "transport_error": 2.0,
        "server_error": 2.0,
        "rate_limited": 1.0,
        "auth_expired": 1.0,
        "malformed": 1.0,
        "stale": 2.0,
        "partial": 1.0,
        "empty": 1.0,
        "slow": 1.0,
        "timeout_after_apply": 1.0,
    },
    persistence_weights={
        "permanent": 0.60,
        "intermittent": 0.25,
        "transient": 0.15,
    },
    fault_probability=0.55,
    intermittent_probability=0.50,
)

PROFILES: dict[str, FaultProfile] = {
    p.name: p
    for p in (FAULT_FREE, TRANSIENT_PROFILE, INTERMITTENT_PROFILE, STRESS_PROFILE)
}


@dataclass
class _ToolFaultState:
    affected: bool = False
    mode: FaultMode | None = None
    persistence: Persistence | None = None
    transient_used: bool = False
    transient_at: int = 0


class FaultSchedule:
    """Per-run fault decisions for every tool, drawn from one seed."""

    def __init__(self, profile: FaultProfile, seed: int) -> None:
        self.profile = profile
        self.seed = seed
        self._rng = random.Random(seed)
        self._states: dict[str, _ToolFaultState] = {}
        self.faulted_calls: list[dict[str, Any]] = []

    def _state(self, tool: str) -> _ToolFaultState:
        state = self._states.get(tool)
        if state is None:
            state = _ToolFaultState()
            if self.profile.fault_probability > 0 and self._rng.random() < (
                self.profile.fault_probability
            ):
                state.affected = True
                state.mode = self.profile.sample_mode(self._rng)
                state.persistence = self.profile.sample_persistence(self._rng)
                # Transient faults strike at a seeded call index (0-2).
                state.transient_at = self._rng.randint(0, 2)
            self._states[tool] = state
        return state

    def decide(self, tool: str, call_index: int) -> FaultMode | None:
        """Return the fault for this call, or ``None`` for a clean call."""
        if self.profile.fault_probability <= 0:
            return None
        state = self._state(tool)
        if not state.affected or state.mode is None:
            return None
        mode: FaultMode | None = None
        if state.persistence == Persistence.PERMANENT:
            mode = state.mode
        elif state.persistence == Persistence.TRANSIENT:
            if call_index == state.transient_at and not state.transient_used:
                mode = state.mode
                state.transient_used = True
        elif state.persistence == Persistence.INTERMITTENT:
            if self._rng.random() < self.profile.intermittent_probability:
                mode = state.mode
        if mode is not None:
            self.faulted_calls.append(
                {
                    "tool": tool,
                    "call_index": call_index,
                    "mode": mode.value,
                    "persistence": state.persistence.value,
                }
            )
        return mode

    def affected_tools(self) -> list[str]:
        return [t for t, s in self._states.items() if s.affected]


class FaultInjector:
    """Middleware between an agent's client and a server.

    ``wrap`` applies the scheduled fault to a clean server result:
    explicit modes raise :class:`ToolError`, corrupt modes return a
    damaged payload, and ``slow`` inflates the latency. The injector
    owns no server state except the schedule; write effects for
    ``timeout_after_apply`` are applied by the server before the
    injector raises, via the ``applied`` flag the server returns.
    """

    def __init__(
        self,
        schedule: FaultSchedule,
        *,
        timeout_ms: int = 2_000,
        stale_age_ms: int = 60_000,
    ) -> None:
        self.schedule = schedule
        self.timeout_ms = timeout_ms
        self.stale_age_ms = stale_age_ms
        self._call_counts: dict[str, int] = {}

    def apply(
        self,
        tool: str,
        result: ToolResult,
        *,
        applied: bool = False,
    ) -> ToolResult:
        call_index = self._call_counts.get(tool, 0)
        self._call_counts[tool] = call_index + 1
        mode = self.schedule.decide(tool, call_index)
        if mode is None:
            return result
        if mode == FaultMode.TIMEOUT:
            raise ToolError("timeout", f"{tool} timed out", self.timeout_ms)
        if mode == FaultMode.TIMEOUT_AFTER_APPLY:
            # The effect (if any) was already applied by the server;
            # only the response is lost. ``applied`` is informational:
            # the server applied the write before returning.
            raise ToolError(
                "timeout_after_apply",
                f"{tool} timed out after applying",
                self.timeout_ms,
            )
        if mode == FaultMode.TRANSPORT_ERROR:
            raise ToolError("transport_error", f"{tool} connection reset", 50)
        if mode == FaultMode.SERVER_ERROR:
            raise ToolError("server_error", f"{tool} returned 500", 120)
        if mode == FaultMode.RATE_LIMITED:
            raise ToolError("rate_limited", f"{tool} returned 429", 80)
        if mode == FaultMode.AUTH_EXPIRED:
            raise ToolError("auth_expired", f"{tool} returned 401", 60)
        if mode == FaultMode.MALFORMED:
            raise ToolError("malformed", f"{tool} returned unparseable payload", 90)
        if mode == FaultMode.SLOW:
            return ToolResult(
                tool=result.tool,
                payload=result.payload,
                latency_ms=result.latency_ms * 5 + 1_500,
                stale=result.stale,
                source=result.source,
            )
        if mode == FaultMode.STALE:
            payload = dict(result.payload)
            payload["as_of"] = payload.get("as_of", 0) - self.stale_age_ms
            # A stale payload carries an outdated value, not just an old
            # timestamp: in this simulation the previous-cycle value is
            # 15% higher than the current one (see docs/RESEARCH.md).
            if isinstance(payload.get("value"), (int, float)):
                payload["value"] = round(float(payload["value"]) * 1.15, 4)
            return ToolResult(
                tool=result.tool,
                payload=payload,
                latency_ms=result.latency_ms,
                stale=True,
                source=result.source,
            )
        if mode == FaultMode.PARTIAL:
            payload = dict(result.payload)
            payload.pop("value", None)
            return ToolResult(
                tool=result.tool,
                payload=payload,
                latency_ms=result.latency_ms,
                stale=result.stale,
                source=result.source,
            )
        if mode == FaultMode.EMPTY:
            return ToolResult(
                tool=result.tool,
                payload={},
                latency_ms=result.latency_ms,
                stale=result.stale,
                source=result.source,
            )
        raise AssertionError(f"unhandled fault mode {mode}")  # pragma: no cover
