#!/usr/bin/env python3
"""Validate the t_k=k inner-iteration schedule against the flat T_in schedule
and against solving each selected block's subproblem exactly once (T_in=1),
on a small synthetic sparse-UOT instance -- small enough that t_k=k (O(K^2)
total inner iterations) is tractable.

This does not reproduce publication-scale numbers; it validates the
qualitative story: under a schedule the paper's own convergence remark
(right after Theorem theorem:convergence: t_k=k inner iterations of an
O(1/t)-rate oracle gives eps_k <= C/k) actually endorses, does the
block-coordinate method still reach a matched target accuracy more cheaply
(in matvec-pass-equivalents) than the full-dimensional Bregman-proximal DCA
baseline?

Usage:
    python3 run_schedule_comparison.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sparse_ot_core as core

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "schedule_comparison"


def build_small_problem(seed: int = 0) -> core.SparseOTProblem:
    return core.make_clustered_problem(
        num_source=48,
        num_target=96,
        dimension=8,
        top_k=2,
        source_kl_weight=50.0,
        target_kl_weight=50.0,
        quadratic_weight=0.5,
        sparsity_weight=1.0,
        seed=seed,
    )


def run_condition(
    problem: core.SparseOTProblem,
    method_key: str,
    label: str,
    num_sweeps: int,
    *,
    use_inner_iterations: bool,
    max_inner_iterations: int = 1,
    inner_iterations_schedule=None,
    inner_tol: float = 1e-12,
    record_every_sweeps: int = 1,
    seed: int = 0,
    inner_usage_log=None,
) -> pd.DataFrame:
    config = core.SolverConfig(
        num_sweeps=num_sweeps,
        selection_rule=method_key,
        candidate_batch_size=8,
        seed=seed,
        record_every_sweeps=record_every_sweeps,
        sampling="random_reshuffling",
        max_inner_iterations=max_inner_iterations,
        inner_tol=inner_tol,
    )
    t0 = time.perf_counter()
    plan, history = core.solve_bcdc(
        problem,
        config,
        use_inner_iterations=use_inner_iterations,
        inner_iterations_schedule=inner_iterations_schedule,
        inner_usage_log=inner_usage_log,
    )
    elapsed = time.perf_counter() - t0
    history = history.copy()
    history["condition"] = label
    history["label"] = history["method"] + f" [{label}]"
    final_obj = float(core.objective_value(problem, plan))
    print(f"  {label:>28s} | {method_key:>12s} | final obj={final_obj:10.4f} | "
          f"matvec={history['matvec_pass_equivalent'].iloc[-1]:10.1f} | {elapsed/60:5.2f} min")
    return history


def run_full_baseline(problem: core.SparseOTProblem, num_sweeps: int, seed: int = 0) -> pd.DataFrame:
    config = core.SolverConfig(
        num_sweeps=num_sweeps,
        selection_rule="uniform",  # unused by solve_full_dca, required by validate()
        candidate_batch_size=8,
        seed=seed,
        record_every_sweeps=1,
        max_inner_iterations=100,
        inner_tol=1e-9,
    )
    t0 = time.perf_counter()
    plan, history = core.solve_full_dca(problem, config, geometry="entropy")
    elapsed = time.perf_counter() - t0
    history = history.copy()
    history["condition"] = "n/a (full-dim baseline)"
    history["label"] = history["method"] + " [full-dim baseline]"
    final_obj = float(core.objective_value(problem, plan))
    print(f"  {'Full non-Euclidean DCA':>28s} | {'full_entropy':>12s} | final obj={final_obj:10.4f} | "
          f"matvec={history['matvec_pass_equivalent'].iloc[-1]:10.1f} | {elapsed/60:5.2f} min")
    return history


def report(combined: pd.DataFrame, tolerances_pct: list[float]) -> None:
    obj_star = combined["objective"].min()
    print(f"\nShared target objective (best across all conditions): {obj_star:.6f}\n")

    print("=== Final objective + total matvec cost per condition ===")
    finals = (
        combined.sort_values("matvec_pass_equivalent")
        .groupby("label")
        .tail(1)[["label", "objective", "matvec_pass_equivalent"]]
        .sort_values("objective")
    )
    print(finals.to_string(index=False))
    print()

    print("=== Matvec-pass-equivalent cost to reach within X% of the shared target ===")
    for tol in tolerances_pct:
        threshold = obj_star * (1.0 + tol / 100.0)
        rows = []
        for label, g in combined.groupby("label"):
            g = g.sort_values("matvec_pass_equivalent")
            hit = g[g["objective"] <= threshold]
            cost = hit["matvec_pass_equivalent"].iloc[0] if len(hit) else float("nan")
            rows.append({"label": label, "matvec_to_reach": cost, "reached": len(hit) > 0})
        sub = pd.DataFrame(rows).sort_values("matvec_to_reach")
        print(f"--- within {tol}% of best ({threshold:.2f}) ---")
        print(sub[["label", "matvec_to_reach", "reached"]].to_string(index=False))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_small_problem(seed=args.seed)
    print(f"Problem: num_source={problem.num_source}, num_target={problem.num_target}\n")

    histories = []
    for method_key, name in [("uniform", "Randomized-BCDC"), ("bregman_gap", "GS-gap-BCDC")]:
        print(f"[{name}]")
        histories.append(
            run_condition(
                problem, method_key, "T_in=1", num_sweeps=1500,
                use_inner_iterations=False, max_inner_iterations=1,
                record_every_sweeps=5, seed=args.seed,
            )
        )
        histories.append(
            run_condition(
                problem, method_key, "flat T_in=50", num_sweeps=150,
                use_inner_iterations=True, max_inner_iterations=50,
                record_every_sweeps=2, seed=args.seed,
            )
        )
        histories.append(
            run_condition(
                problem, method_key, "t_k=k schedule", num_sweeps=50,
                use_inner_iterations=True,
                inner_iterations_schedule=lambda k: k,
                record_every_sweeps=1, seed=args.seed,
            )
        )
        print()

    print("[Full non-Euclidean DCA baseline]")
    histories.append(run_full_baseline(problem, num_sweeps=300, seed=args.seed))
    print()

    combined = pd.concat(histories, ignore_index=True)
    combined.to_csv(args.output_dir / "raw_histories.csv", index=False)
    report(combined, tolerances_pct=[20.0, 10.0, 5.0, 1.0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
