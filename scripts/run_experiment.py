"""Run the full experiment grid and write results/*.json.

Reproducibility: the grid is a pure function of (base_seed, n_seeds).
Run it twice and the two ``experiment.json`` files are identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_fault_harness.experiment import run_grid, write_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    summaries = run_grid(n_seeds=args.seeds, base_seed=args.seed)
    measured = write_results(summaries, args.out_dir)
    measured["n_seeds_per_task"] = args.seeds
    measured["base_seed"] = args.seed
    (args.out_dir / "measured.json").write_text(json.dumps(measured, indent=2) + "\n")
    for summary in summaries:
        print(
            f"{summary.profile}/{summary.policy}/{summary.capability}: "
            f"score={summary.mean_score:.2f} grounded={summary.mean_grounded_score:.2f} "
            f"resilience={summary.resilience_ratio:.3f} "
            f"completion={summary.completion_rate:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
