"""Small end-to-end demo: one chain task under the stress profile.

Shows, step by step, how each policy handles the same seeded faults:
the naive policy accepts a corrupt payload, retry re-tries a dead
tool, fallback serves a stale value, and the resilient policy
abstains rather than guessing when nothing verified remains.
"""

from __future__ import annotations

import argparse

from mcp_fault_harness.cli import cmd_single


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for policy in ("naive", "retry", "fallback", "resilient"):
        namespace = argparse.Namespace(
            task="discount_chain",
            policy=policy,
            capability="large",
            profile="stress",
            seed=args.seed,
        )
        print(f"--- policy={policy} ---")
        cmd_single(namespace)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
