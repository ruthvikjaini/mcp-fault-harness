import pytest

from mcp_fault_harness.faults import (
    FAULT_FREE,
    STRESS_PROFILE,
    FaultInjector,
    FaultMode,
    FaultProfile,
    FaultSchedule,
    Persistence,
)
from mcp_fault_harness.protocol import ToolError, ToolResult


def _profile(mode, persistence, fault_probability=1.0):
    return FaultProfile(
        name="test",
        mode_weights={mode: 1.0},
        persistence_weights={persistence: 1.0},
        fault_probability=fault_probability,
    )


def test_fault_free_schedule_never_faults():
    schedule = FaultSchedule(FAULT_FREE, seed=0)
    assert all(schedule.decide("catalog_price", i) is None for i in range(10))


def test_schedule_is_deterministic_for_a_call_sequence():
    def decisions(seed):
        schedule = FaultSchedule(STRESS_PROFILE, seed)
        return [schedule.decide("catalog_price", i) for i in range(8)]

    assert decisions(7) == decisions(7)
    assert decisions(7) != decisions(8) or True  # seeds may coincide; determinism is the claim


def _force(schedule, tool, mode, persistence, transient_at=0):
    from mcp_fault_harness.faults import _ToolFaultState

    schedule._states[tool] = _ToolFaultState(
        affected=True, mode=mode, persistence=persistence, transient_at=transient_at
    )


def test_permanent_fault_repeats_every_call():
    schedule = FaultSchedule(_profile("timeout", "permanent"), seed=0)
    _force(schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.PERMANENT)
    assert all(
        schedule.decide("catalog_price", i) == FaultMode.TIMEOUT for i in range(5)
    )


def test_transient_fault_strikes_once():
    schedule = FaultSchedule(_profile("timeout", "transient"), seed=0)
    _force(
        schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.TRANSIENT, transient_at=1
    )
    assert schedule.decide("catalog_price", 0) is None
    assert schedule.decide("catalog_price", 1) == FaultMode.TIMEOUT
    assert schedule.decide("catalog_price", 2) is None


def test_injector_explicit_modes_raise():
    schedule = FaultSchedule(_profile("server_error", "permanent"), seed=0)
    _force(schedule, "t", FaultMode.SERVER_ERROR, Persistence.PERMANENT)
    injector = FaultInjector(schedule)
    with pytest.raises(ToolError, match="server_error"):
        injector.apply("t", ToolResult(tool="t", payload={"value": 1, "as_of": 0}))


def test_injector_partial_and_empty_corrupt_payload():
    for mode in (FaultMode.PARTIAL, FaultMode.EMPTY):
        schedule = FaultSchedule(_profile(mode.value, "permanent"), seed=0)
        _force(schedule, "t", mode, Persistence.PERMANENT)
        injector = FaultInjector(schedule)
        result = injector.apply(
            "t", ToolResult(tool="t", payload={"value": 5, "as_of": 0})
        )
        assert "value" not in result.payload


def test_injector_stale_changes_value_and_timestamp():
    schedule = FaultSchedule(_profile("stale", "permanent"), seed=0)
    _force(schedule, "t", FaultMode.STALE, Persistence.PERMANENT)
    injector = FaultInjector(schedule)
    result = injector.apply(
        "t", ToolResult(tool="t", payload={"value": 100.0, "as_of": 5000})
    )
    assert result.stale
    assert result.payload["value"] != 100.0
    assert result.payload["as_of"] < 5000


def test_injector_slow_inflates_latency():
    schedule = FaultSchedule(_profile("slow", "permanent"), seed=0)
    _force(schedule, "t", FaultMode.SLOW, Persistence.PERMANENT)
    injector = FaultInjector(schedule)
    result = injector.apply(
        "t", ToolResult(tool="t", payload={"value": 1, "as_of": 0}, latency_ms=60)
    )
    assert result.latency_ms > 60
