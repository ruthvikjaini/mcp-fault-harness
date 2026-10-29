"""The experiment grid: policies x capabilities x fault conditions.

Every cell is ``n_seeds`` replicates of every task. Servers, fault
schedules, and agent RNG streams are rebuilt per run from the run
seed, so the whole grid is reproducible from ``(base_seed,
n_seeds)`` alone. The fallback cache is seeded once from a clean
server set — it is a read-only snapshot, so sharing it across runs is
safe and keeps the grid fast.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from mcp_fault_harness.agents import (
    CAPABILITIES,
    POLICIES,
    Capability,
    Policy,
    ResilientClient,
    SimulatedAgent,
)
from mcp_fault_harness.faults import PROFILES, FaultInjector, FaultProfile, FaultSchedule
from mcp_fault_harness.resilience import FallbackCache
from mcp_fault_harness.scoring import RunScores, score_run
from mcp_fault_harness.servers import build_server_set, find_server
from mcp_fault_harness.tasks import TASKS, Task


@dataclass
class CellSummary:
    profile: str
    policy: str
    capability: str
    n_runs: int
    mean_score: float
    mean_grounded_score: float
    completion_rate: float
    resilience_ratio: float
    grounded_resilience_ratio: float
    guessed_rate: float
    corrupt_rate: float
    abstain_rate: float
    fallback_rate: float
    duplicate_write_rate: float
    mean_calls: float
    mean_retries: float
    mean_fallbacks: float
    mean_short_circuits: float
    mean_virtual_ms: float
    mean_score_by_kind: dict[str, float] = field(default_factory=dict)


def build_fallback_cache() -> FallbackCache:
    """Seed the cache with canonical values for every read step.

    Values are taken from a clean server set, with ``as_of`` rewound
    to 0 so every cached entry is stale by the freshness window. In
    this static dataset the cached values equal the current values;
    the 0.6 fallback credit in scoring is the penalty for unverified
    freshness, not for being wrong. Write steps are not cached: there
    is no safe cached answer to a side effect.
    """
    cache = FallbackCache()
    servers = build_server_set()
    true_values: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        values: dict[str, Any] = {}
        for step in task.steps:
            if step.is_write:
                values[step.id] = step.expected
                continue
            arguments: dict[str, Any] = {}
            for name, raw in step.args.items():
                if isinstance(raw, dict) and "ref" in raw:
                    arguments[name] = values[raw["ref"]]
                else:
                    arguments[name] = raw
            server = find_server(servers, step.tool)
            result, _applied = server.call(step.tool, arguments, now_ms=0)
            payload = dict(result.payload)
            payload["as_of"] = 0
            cache.put(FallbackCache.key(step.tool, arguments), payload)
            values[step.id] = step.expected
        true_values[task.id] = values
    return cache


def run_one(
    task: Task,
    policy: Policy,
    capability: Capability,
    profile: FaultProfile,
    seed: int,
    cache: FallbackCache,
):
    servers = build_server_set()
    schedule = FaultSchedule(profile, seed)
    injector = FaultInjector(schedule)
    client = ResilientClient(servers, injector, cache)
    agent = SimulatedAgent(client, capability, random.Random(seed + 17))
    record = agent.run(task, policy, seed=seed)
    return record


def aggregate(
    profile: str,
    policy: str,
    capability: str,
    scored: list[tuple[Any, RunScores]],
    baseline_score: float,
    baseline_grounded: float,
) -> CellSummary:
    scores = [s.score for _r, s in scored]
    grounded = [s.grounded_score for _r, s in scored]
    by_kind: dict[str, list[float]] = {}
    write_runs = [s for r, s in scored if r.kind == "write"]
    for record, score in scored:
        by_kind.setdefault(record.kind, []).append(score.score)
    return CellSummary(
        profile=profile,
        policy=policy,
        capability=capability,
        n_runs=len(scored),
        mean_score=mean(scores),
        mean_grounded_score=mean(grounded),
        completion_rate=mean(1.0 if s.complete else 0.0 for _r, s in scored),
        resilience_ratio=(mean(scores) / baseline_score) if baseline_score else 0.0,
        grounded_resilience_ratio=(
            (mean(grounded) / baseline_grounded) if baseline_grounded else 0.0
        ),
        guessed_rate=mean([s.guessed_rate for _r, s in scored]),
        corrupt_rate=mean([s.corrupt_rate for _r, s in scored]),
        abstain_rate=mean([s.abstain_rate for _r, s in scored]),
        fallback_rate=mean([s.fallback_rate for _r, s in scored]),
        duplicate_write_rate=(
            mean([1.0 if s.duplicate_write else 0.0 for s in write_runs])
            if write_runs
            else 0.0
        ),
        mean_calls=mean([r.calls for r, _s in scored]),
        mean_retries=mean([r.retries for r, _s in scored]),
        mean_fallbacks=mean([r.fallbacks_used for r, _s in scored]),
        mean_short_circuits=mean([r.short_circuits for r, _s in scored]),
        mean_virtual_ms=mean([r.virtual_ms for r, _s in scored]),
        mean_score_by_kind={k: mean(v) for k, v in sorted(by_kind.items())},
    )


def run_grid(n_seeds: int = 25, base_seed: int = 0) -> list[CellSummary]:
    """Run the full grid and return one summary per cell."""
    cache = build_fallback_cache()
    raw: dict[tuple[str, str, str], list[tuple[Any, RunScores]]] = {}
    for profile_name, profile in PROFILES.items():
        for policy in POLICIES.values():
            for capability in CAPABILITIES.values():
                key = (profile_name, policy.name, capability.name)
                runs: list[tuple[Any, RunScores]] = []
                for replicate in range(n_seeds):
                    for task_index, task in enumerate(TASKS):
                        seed = base_seed + replicate * 1000 + task_index * 131
                        record = run_one(task, policy, capability, profile, seed, cache)
                        runs.append((record, score_run(record)))
                raw[key] = runs
    summaries: list[CellSummary] = []
    for (profile_name, policy_name, capability_name), runs in raw.items():
        baseline = raw[("fault_free", policy_name, capability_name)]
        baseline_score = mean([s.score for _r, s in baseline])
        baseline_grounded = mean([s.grounded_score for _r, s in baseline])
        summaries.append(
            aggregate(profile_name, policy_name, capability_name, runs, baseline_score, baseline_grounded)
        )
    return summaries


def summaries_to_json(summaries: list[CellSummary]) -> dict[str, Any]:
    return {
        "cells": [asdict(s) for s in summaries],
    }


def write_results(summaries: list[CellSummary], out_dir: Path) -> dict[str, Any]:
    """Write ``experiment.json`` (full grid) and ``measured.json`` (summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = summaries_to_json(summaries)
    (out_dir / "experiment.json").write_text(json.dumps(full, indent=2) + "\n")

    def cell(profile: str, policy: str, capability: str) -> CellSummary:
        return next(
            s
            for s in summaries
            if s.profile == profile and s.policy == policy and s.capability == capability
        )

    measured = {
        "n_seeds_per_task": None,  # filled by the caller script
        "cells": {f"{s.profile}/{s.policy}/{s.capability}": asdict(s) for s in summaries},
        "headline": {
            "stress_naive_small": asdict(cell("stress", "naive", "small")),
            "stress_naive_large": asdict(cell("stress", "naive", "large")),
            "stress_resilient_small": asdict(cell("stress", "resilient", "small")),
            "stress_resilient_large": asdict(cell("stress", "resilient", "large")),
            "fault_free_resilient_large": asdict(cell("fault_free", "resilient", "large")),
            "fault_free_naive_large": asdict(cell("fault_free", "naive", "large")),
        },
    }
    (out_dir / "measured.json").write_text(json.dumps(measured, indent=2) + "\n")
    return measured
