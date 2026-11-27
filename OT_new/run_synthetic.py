#!/usr/bin/env python3
"""Monte Carlo sparse-UOT benchmark on a clustered synthetic instance."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import sparse_ot_core as core
from experiment_io import (
    BASE_METRICS,
    aggregate_outputs,
    make_run_config,
    paper_curve,
    publish_paper_outputs,
    run_methods,
    write_metadata,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "synthetic_mc10"


def configure_experiment(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(core.CONFIG)
    config["record_every_sweeps"] = int(args.record_every_sweeps)
    outer = copy.deepcopy(config["outer_iterations_by_method"])
    if args.uniform_sweeps is not None:
        outer["uniform"] = int(args.uniform_sweeps)
    if args.gs_sweeps is not None:
        outer["lipschitz"] = int(args.gs_sweeps)
        outer["bregman_gap"] = int(args.gs_sweeps)
    if args.full_outer_iterations is not None:
        outer["full_entropy"] = int(args.full_outer_iterations)
        outer["full_euclidean"] = int(args.full_outer_iterations)
    config["outer_iterations_by_method"] = outer

    inner = copy.deepcopy(config["max_inner_iterations_by_method"])
    if args.full_inner_iterations is not None:
        inner["full_entropy"] = int(args.full_inner_iterations)
        inner["full_euclidean"] = int(args.full_inner_iterations)
    config["max_inner_iterations_by_method"] = inner

    return config


def build_problem(config: dict) -> core.SparseOTProblem:
    return core.make_clustered_problem(
        num_source=config["num_source"],
        num_target=config["num_target"],
        dimension=config["dimension"],
        top_k=config["top_k"],
        source_kl_weight=config["source_kl_weight"],
        target_kl_weight=config["target_kl_weight"],
        quadratic_weight=config["quadratic_weight"],
        sparsity_weight=config["sparsity_weight"],
        target_mass_log_std=config["target_mass_log_std"],
        outlier_fraction=config["outlier_fraction"],
        outlier_shift=config["outlier_shift"],
        noise_std=config["noise_std"],
        seed=config["problem_seed"],
    )


def write_paper_outputs(output_dir: Path, aggregated) -> Path:
    paper_dir = output_dir / "paper_csv"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_curve(
        aggregated.objective_vs_matvec,
        "matvec_pass_equivalent",
        "objective",
        "objective",
    ).to_csv(paper_dir / "convergence_sparse_ot__objective.csv", index=False)
    paper_curve(
        aggregated.feasibility_vs_matvec,
        "matvec_pass_equivalent",
        "marginal_relative_l1_residual",
        "feasibility",
    ).to_csv(paper_dir / "convergence_sparse_ot__feasibility.csv", index=False)
    return paper_dir


def aggregate_and_publish(args: argparse.Namespace):
    aggregated = aggregate_outputs(
        args.output_dir,
        metric_columns=BASE_METRICS,
        grid_size=args.grid_size,
        include_zero=args.include_zero_grid,
    )
    if aggregated is None:
        return None
    paper_dir = write_paper_outputs(args.output_dir, aggregated)
    if args.paper_data_dir is not None:
        publish_paper_outputs(paper_dir, args.paper_data_dir)
    return aggregated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-data-dir", type=Path)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--record-every-sweeps", type=int, default=1)
    parser.add_argument("--uniform-sweeps", type=int)
    parser.add_argument("--gs-sweeps", type=int)
    parser.add_argument("--full-outer-iterations", type=int)
    parser.add_argument("--full-inner-iterations", type=int)
    parser.add_argument("--include-zero-grid", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if not 0 <= args.run_start <= args.num_runs:
        raise ValueError("--run-start must be between zero and --num-runs.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.record_every_sweeps <= 0:
        raise ValueError("--record-every-sweeps must be positive.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config = configure_experiment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        args.output_dir / "monte_carlo_metadata.json",
        {
            "experiment": "Synthetic clustered sparse UOT",
            "core_module": "sparse_ot_core.py",
            "num_runs_requested": args.num_runs,
            "run_start": args.run_start,
            "grid_size": args.grid_size,
            "include_zero_grid": args.include_zero_grid,
            "base_config": config,
        },
    )

    core.make_method_budget_table(
        core.resolve_method_integer_budgets(
            config.get("outer_iterations_by_method"),
            default=config["num_sweeps"],
            name="outer_iterations_by_method",
        ),
        core.resolve_method_integer_budgets(
            config.get("max_inner_iterations_by_method"),
            default=config["max_inner_iterations"],
            name="max_inner_iterations_by_method",
        ),
        int(config["num_target"]),
    ).to_csv(args.output_dir / "method_budget_table.csv", index=False)

    started = time.perf_counter()
    for run_index in range(args.run_start, args.num_runs):
        raw_path = args.output_dir / f"raw_histories_run_{run_index:03d}.csv"
        final_path = args.output_dir / f"final_run_{run_index:03d}.csv"
        print(f"[run {run_index:02d}] starting", flush=True)
        run_config = make_run_config(config, run_index)
        raw, final = run_methods(
            build_problem(run_config), run_config, run_index=run_index
        )
        raw.to_csv(raw_path, index=False)
        final.to_csv(final_path, index=False)
        aggregate_and_publish(args)
        print(
            f"[run {run_index:02d}] saved. Total elapsed "
            f"{(time.perf_counter() - started) / 60.0:.2f} min.",
            flush=True,
        )

    aggregate_and_publish(args)
    print(f"Done. CSV outputs are in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
