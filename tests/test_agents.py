import random

from mcp_fault_harness.agents import (
    FALLBACK,
    LARGE,
    NAIVE,
    RESILIENT,
    RETRY,
    PROVENANCE_ABSTAINED,
    PROVENANCE_CORRUPT,
    PROVENANCE_FALLBACK,
    PROVENANCE_GROUNDED,
    ResilientClient,
    SimulatedAgent,
)
from mcp_fault_harness.experiment import build_fallback_cache
from mcp_fault_harness.faults import (
    FAULT_FREE,
    FaultInjector,
    FaultMode,
    FaultProfile,
    FaultSchedule,
    Persistence,
    _ToolFaultState,
)
from mcp_fault_harness.scoring import score_run
from mcp_fault_harness.servers import build_server_set, find_server
from mcp_fault_harness.tasks import TASKS


def _task(task_id):
    return next(t for t in TASKS if t.id == task_id)


def _setup(profile, seed=0):
    servers = build_server_set()
    schedule = FaultSchedule(profile, seed)
    injector = FaultInjector(schedule)
    cache = build_fallback_cache()
    client = ResilientClient(servers, injector, cache)
    return servers, schedule, client


def _force(schedule, tool, mode, persistence, transient_at=0):
    schedule._states[tool] = _ToolFaultState(
        affected=True, mode=mode, persistence=persistence, transient_at=transient_at
    )


def _profile(mode, persistence):
    return FaultProfile(
        name="test",
        mode_weights={mode: 1.0},
        persistence_weights={persistence: 1.0},
        fault_probability=1.0,
    )


def test_fault_free_large_resilient_scores_full():
    _servers, _schedule, client = _setup(FAULT_FREE)
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), RESILIENT, seed=0)
    assert score_run(record).score == 10.0
    assert record.outcomes[0].provenance == PROVENANCE_GROUNDED


def test_naive_accepts_partial_payload_as_corrupt():
    _servers, schedule, client = _setup(_profile("partial", "permanent"))
    _force(schedule, "catalog_price", FaultMode.PARTIAL, Persistence.PERMANENT)
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), NAIVE, seed=0)
    assert record.outcomes[0].provenance == PROVENANCE_CORRUPT
    assert score_run(record).score == 0.0


def test_retry_recovers_from_transient_timeout():
    _servers, schedule, client = _setup(_profile("timeout", "transient"))
    _force(
        schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.TRANSIENT, transient_at=0
    )
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), RETRY, seed=0)
    assert record.outcomes[0].provenance == PROVENANCE_GROUNDED
    assert record.retries == 1


def test_resilient_abstains_under_permanent_fault():
    _servers, schedule, client = _setup(_profile("timeout", "permanent"))
    _force(schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.PERMANENT)
    # Remove the fallback entry so abstention, not fallback, is exercised.
    client.cache.entries.clear()
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), RESILIENT, seed=0)
    assert record.outcomes[0].provenance == PROVENANCE_ABSTAINED
    assert not record.complete


def test_fallback_serves_stale_value_at_partial_credit():
    _servers, schedule, client = _setup(_profile("timeout", "permanent"))
    _force(schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.PERMANENT)
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), FALLBACK, seed=0)
    assert record.outcomes[0].provenance == PROVENANCE_FALLBACK
    assert score_run(record).score == 6.0


def test_corrupt_first_step_cascades_through_chain():
    _servers, schedule, client = _setup(_profile("partial", "permanent"))
    _force(schedule, "catalog_price", FaultMode.PARTIAL, Persistence.PERMANENT)
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("discount_chain"), NAIVE, seed=0)
    assert record.outcomes[0].provenance == PROVENANCE_CORRUPT
    # Downstream steps computed from the corrupt value are wrong too.
    assert all(not o.correct for o in record.outcomes)


def test_idempotency_prevents_duplicate_on_timeout_after_apply():
    servers, schedule, client = _setup(_profile("timeout_after_apply", "transient"))
    _force(
        schedule,
        "place_order",
        FaultMode.TIMEOUT_AFTER_APPLY,
        Persistence.TRANSIENT,
        transient_at=0,
    )
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("place_order"), RESILIENT, seed=0)
    server = find_server(servers, "place_order")
    assert len(server.effects) == 1
    assert not record.outcomes[0].duplicate_effect


def test_retry_without_idempotency_duplicates_on_timeout_after_apply():
    servers, schedule, client = _setup(_profile("timeout_after_apply", "transient"))
    _force(
        schedule,
        "place_order",
        FaultMode.TIMEOUT_AFTER_APPLY,
        Persistence.TRANSIENT,
        transient_at=0,
    )
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("place_order"), RETRY, seed=0)
    server = find_server(servers, "place_order")
    assert len(server.effects) == 2
    assert record.outcomes[0].duplicate_effect
    assert score_run(record).score == 0.0


def test_breaker_short_circuits_permanent_fault():
    _servers, schedule, client = _setup(_profile("timeout", "permanent"))
    _force(schedule, "catalog_price", FaultMode.TIMEOUT, Persistence.PERMANENT)
    client.cache.entries.clear()
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    # Run the same failing task repeatedly through one client: after
    # three consecutive failures the breaker opens and calls stop.
    for _ in range(6):
        agent.run(_task("price_lookup"), RESILIENT, seed=0)
    assert client.short_circuits > 0
    assert client.calls < 6 * (1 + RESILIENT.max_retries)


def test_auth_failure_is_not_retried():
    _servers, schedule, client = _setup(_profile("auth_expired", "permanent"))
    _force(schedule, "catalog_price", FaultMode.AUTH_EXPIRED, Persistence.PERMANENT)
    agent = SimulatedAgent(client, LARGE, random.Random(0))
    record = agent.run(_task("price_lookup"), RETRY, seed=0)
    assert record.retries == 0
    assert record.calls == 1
