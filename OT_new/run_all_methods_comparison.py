#!/usr/bin/env python3
"""Check whether GS-gap-BCDC at its sweet-spot T_in becomes the outright
cheapest method among the full 5-method comparison set used in the paper's
CIFAR-10 figure (RBDC-KL, RBDC-KL-BGS, BDC-KL, RBDC-KL-EGS, BDC-Euclidean),
on the same small synthetic instance -- i.e. does it reproduce the PIP story
where the flagship coordinate+Bregman-gap-selection method dominates at
every accuracy threshold, not just some.

Reuses already-saved histories (Randomized-BCDC T_in=1, GS-gap-BCDC T_in=10
and T_in=20, Full-NE-DCA baseline) instead of rerunning them; only runs what
is missing: GS-gap-BCDC at the requested additional T_in values,
Full-Euclidean-DCA, and GS-Lipschitz-BCDC.

Usage:
    python3 run_all_methods_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import run_schedule_comparison as base
import sparse_ot_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEDULE_HIST = SCRIPT_DIR / "outputs" / "schedule_comparison" / "raw_histories.csv"
SWEEP_HIST = SCRIPT_DIR / "outputs" / "tin_sweep" / "raw_histories.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "all_methods_comparison"

# GS-gap-BCDC T_in values to compare (sweet-spot candidates from the sweep).
GS_GAP_SWEET_SPOT_TIN = [10, 15, 20]


def load_reused(path: Path, labels_wanted: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["label"].isin(labels_wanted)].copy()


def run_full_euclidean(problem: core.SparseOTProblem, num_sweeps: int, seed: int = 0) -> pd.DataFrame:
    config = core.SolverConfig(
        num_sweeps=num_sweeps, selection_rule="uniform", candidate_batch_size=8,
        seed=seed, record_every_sweeps=1, max_inner_iterations=100, inner_tol=1e-9,
    )
    plan, history = core.solve_full_dca(problem, config, geometry="euclidean")
    history = history.copy()
    history["condition"] = "n/a (full-dim baseline)"
    history["label"] = history["method"] + " [full-dim baseline]"
    print(f"  {'Full Euclidean DCA':>28s} | final obj={float(core.objective_value(problem, plan)):10.4f} | "
          f"matvec={history['matvec_pass_equivalent'].iloc[-1]:10.1f}")
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = base.build_small_problem(seed=args.seed)
    print(f"Problem: num_source={problem.num_source}, num_target={problem.num_target}\n")

    histories = []

    print("[Reusing saved histories]")
    histories.append(load_reused(SCHEDULE_HIST, ["Randomized entropy-BCDC [T_in=1]"]))
    histories.append(load_reused(SCHEDULE_HIST, ["Full non-Euclidean DCA [full-dim baseline]"]))
    histories.append(load_reused(
        SWEEP_HIST,
        [f"GS-non-Euclidean-gap entropy-BCDC [T_in={t}]" for t in GS_GAP_SWEET_SPOT_TIN if t in (10, 20)],
    ))
    for h in histories:
        for label in h["label"].unique():
            print(f"  reused: {label}")
    print()

    print("[New runs]")
    missing_tin = [t for t in GS_GAP_SWEET_SPOT_TIN if t not in (10, 20)]
    for tin in missing_tin:
        # Same budget curve as the earlier sweep (interpolated for tin=15).
        num_sweeps = 320
        histories.append(
            base.run_condition(
                problem, "bregman_gap", f"T_in={tin}", num_sweeps=num_sweeps,
                use_inner_iterations=True, max_inner_iterations=tin,
                record_every_sweeps=2, seed=args.seed,
            )
        )

    histories.append(run_full_euclidean(problem, num_sweeps=300, seed=args.seed))

    histories.append(
        base.run_condition(
            problem, "lipschitz", "T_in=1", num_sweeps=1500,
            use_inner_iterations=False, max_inner_iterations=1,
            record_every_sweeps=5, seed=args.seed,
        )
    )
    print()

    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(args.output_dir / "raw_histories.csv", index=False)
    base.report(combined, tolerances_pct=[20.0, 10.0, 5.0, 1.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
