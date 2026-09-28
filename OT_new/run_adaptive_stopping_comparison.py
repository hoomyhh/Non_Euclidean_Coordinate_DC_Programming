#!/usr/bin/env python3
"""Compare adaptive inner-loop stopping (repeat the Bregman-proximal column
solve, re-linearizing at each step, stopping once the relative change in the
TRUE column-restricted phi_hat drops below inner_tol, capped at 50 inner
iterations as a safety net -- matching PIP's own "relative inner-objective
tolerance" convention) against the three fixed schedules already tested:
T_in=1, flat T_in=50, and the growing t_k=k schedule.

Reuses saved T_in=1 / flat T_in=50 / t_k=k histories from
outputs/schedule_comparison/raw_histories.csv instead of rerunning them.
Also reports the distribution of inner-iterations actually used per outer
step under adaptive stopping, via solve_bcdc's inner_usage_log.

Usage:
    python3 run_adaptive_stopping_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_schedule_comparison as base

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEDULE_HIST = SCRIPT_DIR / "outputs" / "schedule_comparison" / "raw_histories.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "adaptive_stopping_comparison"

REUSE_CONDITIONS = ["T_in=1", "flat T_in=50", "t_k=k schedule"]
# 1e-6/1e-8 (as originally suggested) were checked on this problem and found
# to saturate the 50-iteration cap on essentially every outer step -- i.e.
# they reduce to flat T_in=50 rather than stopping early. Swept tolerances
# first (see report) to find the tolerance range where the rule actually
# stops early some of the time but not always: ~1e-2 (≈T_in=1), ~1e-3
# (genuinely mixed, median ~23/50), ~1e-4 (≈flat T_in=50 already).
ADAPTIVE_TOLS = [1e-2, 1e-3, 1e-4]


def print_usage_stats(label: str, usage_log: list[int]) -> None:
    arr = np.asarray(usage_log)
    print(
        f"    inner-iterations per outer step [{label}]: "
        f"n={arr.size}, min={arr.min()}, median={np.median(arr):.1f}, "
        f"mean={arr.mean():.2f}, p90={np.percentile(arr, 90):.1f}, max={arr.max()}"
    )
    # Coarse histogram: how often does it stop at 1, vs use meaningfully more.
    at_one = int((arr == 1).sum())
    at_cap = int((arr == arr.max()).sum()) if arr.max() > 1 else 0
    print(
        f"      stopped at 1 inner step: {at_one}/{arr.size} "
        f"({100*at_one/arr.size:.1f}%); hit the cap ({arr.max()}): "
        f"{at_cap}/{arr.size} ({100*at_cap/arr.size:.1f}%)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = base.build_small_problem(seed=args.seed)
    print(f"Problem: num_source={problem.num_source}, num_target={problem.num_target}\n")

    existing = pd.read_csv(SCHEDULE_HIST)
    reused = existing[existing["condition"].isin(REUSE_CONDITIONS)].copy()
    print(f"Reused {reused['label'].nunique()} method/condition histories from {SCHEDULE_HIST}\n")

    histories = [reused]
    for method_key, name in [("uniform", "Randomized-BCDC"), ("bregman_gap", "GS-gap-BCDC")]:
        print(f"[{name}]")
        for tol in ADAPTIVE_TOLS:
            usage_log: list[int] = []
            label = f"adaptive tol={tol:g}"
            histories.append(
                base.run_condition(
                    problem, method_key, label, num_sweeps=500,
                    use_inner_iterations=True, max_inner_iterations=50,
                    inner_tol=tol, record_every_sweeps=2, seed=args.seed,
                    inner_usage_log=usage_log,
                )
            )
            print_usage_stats(f"{name}, {label}", usage_log)
        print()

    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(args.output_dir / "raw_histories.csv", index=False)
    base.report(combined, tolerances_pct=[20.0, 10.0, 5.0, 1.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
