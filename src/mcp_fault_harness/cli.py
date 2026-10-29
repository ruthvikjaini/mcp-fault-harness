"""Command line entry point.

Usage (from the repository root, with ``PYTHONPATH=src``)::

    python -m mcp_fault_harness.cli demo
    python -m mcp_fault_harness.cli grid --seeds 25
    python -m mcp_fault_harness.cli single --task discount_chain \\
        --policy resilient --capability large --profile stress --seed 0
"""

from __future__ import annotations

import argparse
import json
import random

from mcp_fault_harness.agents import CAPABILITIES, POLICIES, SimulatedAgent, ResilientClient
from mcp_fault_harness.experiment import build_fallback_cache, run_grid, run_one
from mcp_fault_harness.faults import PROFILES, FaultInjector, FaultSchedule
from mcp_fault_harness.scoring import score_run
from mcp_fault_harness.servers import build_server_set
from mcp_fault_harness.tasks import TASKS


def _print_table(summaries) -> None:
    print(f"{'profile':<13}{'policy':<11}{'cap':<7}{'score':>7}{'ground':>8}{'resil.':>8}{'compl.':>8}{'calls':>7}")
    for s in summaries:
        print(
            f"{s.profile:<13}{s.policy:<11}{s.capability:<7}"
            f"{s.mean_score:>7.2f}{s.mean_grounded_score:>8.2f}"
            f"{s.resilience_ratio:>8.3f}{s.completion_rate:>8.3f}{s.mean_calls:>7.2f}"
        )


def cmd_demo(_args: argparse.Namespace) -> int:
    summaries = run_grid(n_seeds=5)
    _print_table([s for s in summaries if s.profile in ("fault_free", "stress")])
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    summaries = run_grid(n_seeds=args.seeds, base_seed=args.seed)
    _print_table(summaries)
    print(json.dumps({"cells": len(summaries)}, indent=2))
    return 0


def cmd_single(args: argparse.Namespace) -> int:
    task = next(t for t in TASKS if t.id == args.task)
    policy = POLICIES[args.policy]
    capability = CAPABILITIES[args.capability]
    profile = PROFILES[args.profile]
    cache = build_fallback_cache()
    record = run_one(task, policy, capability, profile, args.seed, cache)
    scores = score_run(record)
    print(f"task={task.id} kind={task.kind} policy={policy.name} "
          f"capability={capability.name} profile={profile.name} seed={args.seed}")
    for outcome in record.outcomes:
        print(
            f"  {outcome.step_id} {outcome.tool} provenance={outcome.provenance} "
            f"value={outcome.value} expected={outcome.expected} "
            f"correct={outcome.correct} credit={outcome.credit} "
            f"effects={outcome.effect_count}"
        )
    print(f"score={scores.score:.2f} grounded={scores.grounded_score:.2f} "
          f"complete={scores.complete} calls={record.calls} "
          f"retries={record.retries} fallbacks={record.fallbacks_used} "
          f"virtual_ms={record.virtual_ms}")
    print(f"faulted_calls={json.dumps(record.faulted_calls)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mcp-fault-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="small fault-free vs stress comparison")
    grid = sub.add_parser("grid", help="full experiment grid")
    grid.add_argument("--seeds", type=int, default=25)
    grid.add_argument("--seed", type=int, default=0)
    single = sub.add_parser("single", help="run one task and show step outcomes")
    single.add_argument("--task", default="discount_chain", choices=[t.id for t in TASKS])
    single.add_argument("--policy", default="resilient", choices=list(POLICIES))
    single.add_argument("--capability", default="large", choices=list(CAPABILITIES))
    single.add_argument("--profile", default="stress", choices=list(PROFILES))
    single.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "grid":
        return cmd_grid(args)
    return cmd_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
