"""Simulated tool-calling agents and resilience policies.

The agents are *simulated*: they do not call a language model. A
capability level is a set of error rates (how often the agent
misreads a good result, how often a guess happens to be right), and a
policy is a set of client behaviours (validate, retry, fall back,
circuit-break, use idempotency keys, abstain). This separation is the
point of the experiment: capability and handling vary independently,
so a resilience ratio can be computed for each capability level and
compared, the way Iannillo et al. compare the 20B and 120B backbones.

Validation lives in the client, not the agent: a corrupt payload that
fails schema or freshness checks is treated exactly like an explicit
error — retried, then replaced by a fallback, then reported as a
failure. A policy that does not validate accepts the corrupt payload
as evidence, which is the silent-damage family.

One behaviour is deliberately realistic: a step whose arguments come
from an earlier step uses the *observed* value, not the true one. A
guessed, stale, or corrupt value therefore cascades through a chain,
exactly as a bad tool result propagates in a real plan-and-execute
agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mcp_fault_harness.faults import FaultInjector, RETRYABLE_ERROR_KINDS
from mcp_fault_harness.protocol import ToolDefinition, ToolError, ToolResult
from mcp_fault_harness.resilience import (
    CircuitBreaker,
    FallbackCache,
    IdempotencyStore,
)
from mcp_fault_harness.servers import Server, all_definitions, find_server
from mcp_fault_harness.tasks import Step, Task

PROVENANCE_GROUNDED = "grounded"
PROVENANCE_FALLBACK = "fallback"
PROVENANCE_GUESSED = "guessed"
PROVENANCE_ABSTAINED = "abstained"
PROVENANCE_CORRUPT = "corrupt_accepted"


@dataclass(frozen=True)
class Capability:
    """Simulated reasoning capability (the 'model size' axis)."""

    name: str
    base_error_rate: float  # misreads a good result this often
    guess_accuracy: float  # a guess is right this often
    fallback_mishandle_rate: float  # misreads a fallback value this often


SMALL = Capability("small", base_error_rate=0.08, guess_accuracy=0.15, fallback_mishandle_rate=0.10)
LARGE = Capability("large", base_error_rate=0.03, guess_accuracy=0.20, fallback_mishandle_rate=0.05)

CAPABILITIES: dict[str, Capability] = {"small": SMALL, "large": LARGE}


@dataclass(frozen=True)
class Policy:
    """A client-side resilience policy (the 'agent stack' axis)."""

    name: str
    validate: bool
    max_retries: int
    use_fallback: bool
    use_breaker: bool
    use_idempotency: bool
    abstain_on_missing: bool
    breaker_threshold: int = 3
    breaker_reset_ms: int = 1_000


NAIVE = Policy(
    name="naive",
    validate=False,
    max_retries=0,
    use_fallback=False,
    use_breaker=False,
    use_idempotency=False,
    abstain_on_missing=False,
)
RETRY = Policy(
    name="retry",
    validate=True,
    max_retries=2,
    use_fallback=False,
    use_breaker=False,
    use_idempotency=False,
    abstain_on_missing=False,
)
FALLBACK = Policy(
    name="fallback",
    validate=True,
    max_retries=1,
    use_fallback=True,
    use_breaker=False,
    use_idempotency=False,
    abstain_on_missing=False,
)
RESILIENT = Policy(
    name="resilient",
    validate=True,
    max_retries=2,
    use_fallback=True,
    use_breaker=True,
    use_idempotency=True,
    abstain_on_missing=True,
)

POLICIES: dict[str, Policy] = {
    p.name: p for p in (NAIVE, RETRY, FALLBACK, RESILIENT)
}


@dataclass
class StepOutcome:
    step_id: str
    tool: str
    provenance: str
    value: Any
    expected: Any
    correct: bool
    credit: float  # main-score credit, before grounded-score filtering
    effect_count: int = 0
    duplicate_effect: bool = False


@dataclass
class RunRecord:
    task_id: str
    kind: str
    policy: str
    capability: str
    outcomes: list[StepOutcome]
    calls: int
    retries: int
    fallbacks_used: int
    short_circuits: int
    virtual_ms: int
    faulted_calls: list[dict[str, Any]]

    @property
    def complete(self) -> bool:
        return all(o.provenance != PROVENANCE_ABSTAINED for o in self.outcomes)


def wrong_value(expected: Any) -> Any:
    """A plausible-but-wrong value, for guesses and misreads."""
    if isinstance(expected, bool):
        return not expected
    if isinstance(expected, (int, float)):
        return round(float(expected) * 1.25 + 7.0, 4)
    return "wrong"


class ResilientClient:
    """The tool-call path: breaker -> server -> injector -> validation."""

    def __init__(
        self,
        servers: dict[str, Server],
        injector: FaultInjector,
        cache: FallbackCache,
        *,
        timeout_ms: int = 2_000,
    ) -> None:
        self.servers = servers
        self.definitions = all_definitions(servers)
        self.injector = injector
        self.cache = cache
        self.timeout_ms = timeout_ms
        self.now_ms = 0
        self.calls = 0
        self.retries = 0
        self.fallbacks_used = 0
        self.short_circuits = 0
        self._breakers: dict[str, CircuitBreaker] = {}
        self._idempotency = IdempotencyStore()

    def _breaker(self, tool: str, policy: Policy) -> CircuitBreaker:
        breaker = self._breakers.get(tool)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=policy.breaker_threshold,
                reset_timeout_ms=policy.breaker_reset_ms,
            )
            self._breakers[tool] = breaker
        return breaker

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        policy: Policy,
        *,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        """Call a tool under ``policy``. Raises on ultimate failure.

        A cached value is returned as a :class:`ToolResult` with
        ``source="cache"`` and ``stale=True``, never disguised as live.
        """
        definition = self.definitions[tool]
        key = idempotency_key if policy.use_idempotency else None
        if key is not None:
            stored = self._idempotency.get(key)
            if stored is not None:
                return ToolResult(
                    tool=tool, payload=dict(stored), latency_ms=0, source="cache"
                )
        breaker = self._breaker(tool, policy) if policy.use_breaker else None
        attempt = 0
        while True:
            if breaker is not None and not breaker.allow(self.now_ms):
                self.short_circuits += 1
                return self._fallback_or_raise(
                    tool,
                    arguments,
                    policy,
                    ToolError("breaker_open", f"breaker open for {tool}", 0),
                )
            try:
                result = self._single_call(tool, arguments, definition, policy, key)
            except ToolError as error:
                if breaker is not None:
                    breaker.on_failure(self.now_ms)
                if (
                    error.kind in RETRYABLE_ERROR_KINDS
                    and attempt < policy.max_retries
                ):
                    attempt += 1
                    self.retries += 1
                    self.now_ms += 50 * (2 ** (attempt - 1))
                    continue
                return self._fallback_or_raise(tool, arguments, policy, error)
            if breaker is not None:
                breaker.on_success()
            if key is not None:
                self._idempotency.put(key, result.payload)
            return result

    def _fallback_or_raise(
        self,
        tool: str,
        arguments: dict[str, Any],
        policy: Policy,
        error: ToolError,
    ) -> ToolResult:
        if policy.use_fallback:
            cached = self.cache.get(FallbackCache.key(tool, arguments))
            if cached is not None:
                self.fallbacks_used += 1
                payload = dict(cached)
                payload.pop("stale", None)
                return ToolResult(
                    tool=tool,
                    payload=payload,
                    latency_ms=5,
                    stale=True,
                    source="cache",
                )
        raise error

    def _single_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        definition: ToolDefinition,
        policy: Policy,
        key: str | None,
    ) -> ToolResult:
        server = find_server(self.servers, tool)
        result, _applied = server.call(
            tool, arguments, now_ms=self.now_ms, idempotency_key=key
        )
        self.calls += 1
        result = self.injector.apply(tool, result)
        self.now_ms += result.latency_ms
        if result.latency_ms > self.timeout_ms:
            raise ToolError(
                "timeout", f"{tool} exceeded {self.timeout_ms} ms", self.timeout_ms
            )
        if policy.validate:
            valid, reason = definition.validate_payload(
                result.payload, now_ms=self.now_ms
            )
            if not valid:
                raise ToolError("malformed", f"invalid payload: {reason}", result.latency_ms)
        return result


class SimulatedAgent:
    """Runs a task's steps through a client under a policy/capability."""

    def __init__(
        self,
        client: ResilientClient,
        capability: Capability,
        rng: random.Random,
    ) -> None:
        self.client = client
        self.capability = capability
        self.rng = rng

    def run(self, task: Task, policy: Policy, *, seed: int) -> RunRecord:
        values: dict[str, Any] = {}
        outcomes: list[StepOutcome] = []
        for step in task.steps:
            outcome = self._run_step(task, step, policy, values, seed)
            outcomes.append(outcome)
            values[step.id] = outcome.value
        return RunRecord(
            task_id=task.id,
            kind=task.kind,
            policy=policy.name,
            capability=self.capability.name,
            outcomes=outcomes,
            calls=self.client.calls,
            retries=self.client.retries,
            fallbacks_used=self.client.fallbacks_used,
            short_circuits=self.client.short_circuits,
            virtual_ms=self.client.now_ms,
            faulted_calls=list(self.client.injector.schedule.faulted_calls),
        )

    def _resolve_args(
        self, step: Step, values: dict[str, Any]
    ) -> dict[str, Any] | None:
        resolved: dict[str, Any] = {}
        for name, raw in step.args.items():
            if isinstance(raw, dict) and "ref" in raw:
                prior = values.get(raw["ref"])
                if prior is None:
                    return None
                resolved[name] = prior
            else:
                resolved[name] = raw
        return resolved

    def _run_step(
        self,
        task: Task,
        step: Step,
        policy: Policy,
        values: dict[str, Any],
        seed: int,
    ) -> StepOutcome:
        arguments = self._resolve_args(step, values)
        if arguments is None:
            return StepOutcome(
                step_id=step.id,
                tool=step.tool,
                provenance=PROVENANCE_ABSTAINED,
                value=None,
                expected=step.expected,
                correct=False,
                credit=0.0,
            )
        key = f"{task.id}:{step.id}:{seed}" if step.is_write else None
        try:
            result = self.client.call(step.tool, arguments, policy, idempotency_key=key)
        except ToolError:
            outcome = self._missing_evidence(step, policy)
            return self._with_effects(step, arguments, outcome)
        value, provenance = self._interpret(step, result, policy)
        outcome = self._score_outcome(step, value, provenance)
        return self._with_effects(step, arguments, outcome)

    def _with_effects(
        self, step: Step, arguments: dict[str, Any], outcome: StepOutcome
    ) -> StepOutcome:
        if not step.is_write:
            return outcome
        server = find_server(self.client.servers, step.tool)
        effect_count = sum(
            1
            for effect in server.effects
            if effect["tool"] == step.tool and effect["arguments"] == arguments
        )
        duplicate = effect_count > 1
        if duplicate or effect_count == 0:
            outcome.credit = 0.0
            outcome.correct = False
        if outcome.provenance in (PROVENANCE_GUESSED, PROVENANCE_ABSTAINED):
            # A guessed or abstained write is never confirmed: the
            # agent holds no effect id, even if an effect exists.
            outcome.credit = 0.0
            outcome.correct = False
        outcome.effect_count = effect_count
        outcome.duplicate_effect = duplicate
        return outcome

    def _interpret(
        self, step: Step, result: ToolResult, policy: Policy
    ) -> tuple[Any, str]:
        definition = self.client.definitions[step.tool]
        if result.source == "cache":
            if self.rng.random() < self.capability.fallback_mishandle_rate:
                return wrong_value(step.expected), PROVENANCE_FALLBACK
            return result.payload.get("value"), PROVENANCE_FALLBACK
        if not policy.validate:
            # No validation: a payload that fails schema or freshness
            # is still accepted as evidence — silent damage.
            valid, _reason = definition.validate_payload(
                result.payload, now_ms=self.client.now_ms
            )
            if not valid:
                value = result.payload.get("value")
                if value is None:
                    value = wrong_value(step.expected)
                return value, PROVENANCE_CORRUPT
        value = result.payload.get("value")
        if self.rng.random() < self.capability.base_error_rate:
            return wrong_value(step.expected), PROVENANCE_GROUNDED
        return value, PROVENANCE_GROUNDED

    def _missing_evidence(self, step: Step, policy: Policy) -> StepOutcome:
        if policy.abstain_on_missing:
            return StepOutcome(
                step_id=step.id,
                tool=step.tool,
                provenance=PROVENANCE_ABSTAINED,
                value=None,
                expected=step.expected,
                correct=False,
                credit=0.0,
            )
        guessed = self.rng.random() < self.capability.guess_accuracy
        value = step.expected if guessed else wrong_value(step.expected)
        return StepOutcome(
            step_id=step.id,
            tool=step.tool,
            provenance=PROVENANCE_GUESSED,
            value=value,
            expected=step.expected,
            correct=guessed,
            credit=1.0 if guessed else 0.0,
        )

    def _score_outcome(
        self, step: Step, value: Any, provenance: str
    ) -> StepOutcome:
        correct = value == step.expected
        if provenance == PROVENANCE_FALLBACK:
            credit = 0.6 if correct else 0.0
        elif provenance == PROVENANCE_CORRUPT:
            credit = 0.0
            correct = False
        else:
            credit = 1.0 if correct else 0.0
        return StepOutcome(
            step_id=step.id,
            tool=step.tool,
            provenance=provenance,
            value=value,
            expected=step.expected,
            correct=correct,
            credit=credit,
        )
