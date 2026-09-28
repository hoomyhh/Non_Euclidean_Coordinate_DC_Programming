#!/usr/bin/env python3
"""Sweep the flat inner-iteration constant T_in over {1,2,5,10,20,50} on the
same small synthetic instance and matched-accuracy methodology used in
run_schedule_comparison.py, to see where the cost-to-target-accuracy curve
starts rising. Reuses the T_in=1 and T_in=50 histories already saved by
run_schedule_comparison.py (outputs/schedule_comparison/raw_histories.csv)
instead of rerunning those two endpoints.

Usage:
    python3 run_tin_sweep.py [--existing-histories PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import run_schedule_comparison as base
import sparse_ot_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXISTING = SCRIPT_DIR / "outputs" / "schedule_comparison" / "raw_histories.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "tin_sweep"

# T_in -> (num_sweeps, record_every_sweeps), chosen so every value converges
# to roughly the same plateau as the already-validated T_in=1 (1500 sweeps)
# and T_in=50 (150 sweeps) endpoints (log-interpolated between them).
SWEEP_BUDGETS = {
    2: (1000, 5),
    5: (600, 3),
    10: (400, 2),
    20: (250, 1),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-histories", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = base.build_small_problem(seed=args.seed)
    print(f"Problem: num_source={problem.num_source}, num_target={problem.num_target}\n")

    existing = pd.read_csv(args.existing_histories)
    reused = existing[existing["condition"].isin(["T_in=1", "flat T_in=50"])].copy()
    reused["condition"] = reused["condition"].replace(
        {"T_in=1": "T_in=1", "flat T_in=50": "T_in=50"}
    )
    reused["label"] = reused["method"] + " [T_in=" + reused["condition"].str.replace(
        "T_in=", ""
    ) + "]"
    print(f"Reused {reused['label'].nunique()} method/condition histories from {args.existing_histories}\n")

    histories = [reused]
    for method_key, name in [("uniform", "Randomized-BCDC"), ("bregman_gap", "GS-gap-BCDC")]:
        print(f"[{name}]")
        for tin, (num_sweeps, record_every) in SWEEP_BUDGETS.items():
            histories.append(
                base.run_condition(
                    problem, method_key, f"T_in={tin}", num_sweeps=num_sweeps,
                    use_inner_iterations=True, max_inner_iterations=tin,
                    record_every_sweeps=record_every, seed=args.seed,
                )
            )
        print()

    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(args.output_dir / "raw_histories.csv", index=False)
    base.report(combined, tolerances_pct=[20.0, 10.0, 5.0, 1.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
