"""Shared experiment execution and CSV aggregation for sparse UOT."""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import sparse_ot_core as core

BASE_METRICS = (
    "objective",
    "marginal_relative_l1_residual",
    "source_marginal_relative_l1_residual",
    "target_marginal_relative_l1_residual",
    "optimization_time_seconds",
    "wall_clock_time",
    "touched_nonzeros",
    "column_accesses",
    "matvec_pass_equivalent",
)

PAPER_METHOD_LABELS = {
    "uniform": "Randomized-BCDC",
    "bregman_gap": "GS-gap-BCDC",
    "lipschitz": "GS-Lipschitz-BCDC",
    "full_entropy": "Full-NE-DCA",
    "full_euclidean": "Full-Euclidean-DCA",
}


@dataclass(frozen=True)
class AggregatedOutputs:
    raw: pd.DataFrame
    final: pd.DataFrame
    final_summary: pd.DataFrame
    objective_vs_matvec: pd.DataFrame
    objective_vs_time: pd.DataFrame
    feasibility_vs_matvec: pd.DataFrame


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def make_run_config(base_config: dict, run_index: int) -> dict:
    config = copy.deepcopy(base_config)
    config["problem_seed"] = int(base_config["problem_seed"]) + run_index
    config["solver_seed"] = int(base_config["solver_seed"]) + 1009 * run_index
    return config


def build_base_solver_config(config: dict, problem: core.SparseOTProblem):
    return core.SolverConfig(
        num_sweeps=config["num_sweeps"],
        candidate_batch_size=min(config["candidate_batch_size"], problem.num_target),
        seed=config["solver_seed"],
        record_every_sweeps=config["record_every_sweeps"],
        min_plan_value=config["min_plan_value"],
        relative_smoothness_scale=config["relative_smoothness_scale"],
        block_log_radius=config["block_log_radius"],
        sampling="random_reshuffling",
        max_inner_iterations=config["max_inner_iterations"],
        inner_tol=config["inner_tol"],
    )


def run_methods(
    problem: core.SparseOTProblem,
    config: dict,
    run_index: int,
    final_metrics: Callable[[np.ndarray], dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every comparison method and return history and final data frames."""

    initial_plan = core.initialize_plan(problem)
    base_solver_config = build_base_solver_config(config, problem)
    outer_iterations = core.resolve_method_integer_budgets(
        config.get("outer_iterations_by_method"),
        default=config["num_sweeps"],
        name="outer_iterations_by_method",
    )
    inner_iterations = core.resolve_method_integer_budgets(
        config.get("max_inner_iterations_by_method"),
        default=config["max_inner_iterations"],
        name="max_inner_iterations_by_method",
    )

    histories = []
    final_rows = []
    started = time.perf_counter()
    for method_key in core.COMPARISON_METHODS:
        method_started = time.perf_counter()
        solver_config = core.build_method_config(
            base_solver_config, method_key, outer_iterations, inner_iterations
        )
        solver_config.validate(problem.num_target)
        if method_key in core.BCDC_COMPARISON_RULES:
            plan, history = core.solve_bcdc(problem, solver_config, initial_plan)
        elif method_key == "full_entropy":
            plan, history = core.solve_full_dca(
                problem, solver_config, initial_plan, geometry="entropy"
            )
        elif method_key == "full_euclidean":
            plan, history = core.solve_full_dca(
                problem, solver_config, initial_plan, geometry="euclidean"
            )
        else:
            raise ValueError(f"Unknown method: {method_key}")

        history = history.copy()
        history["mc_run"] = int(run_index)
        history["problem_seed"] = int(config["problem_seed"])
        history["solver_seed"] = int(config["solver_seed"])
        history["method_runtime_seconds"] = time.perf_counter() - method_started
        histories.append(history)

        final_row = history.iloc[-1].to_dict()
        if final_metrics is not None:
            final_row.update(final_metrics(plan))
        final_rows.append(final_row)
        print(
            f"[run {run_index:02d}] {core.METHOD_LABELS[method_key]:42s} "
            f"obj={final_row['objective']:.6g} "
            f"feas={final_row['marginal_relative_l1_residual']:.3e} "
            f"matvec={final_row['matvec_pass_equivalent']:.1f}",
            flush=True,
        )

    raw = pd.concat(histories, ignore_index=True)
    final = pd.DataFrame(final_rows)
    best_final = float(final["objective"].min())
    raw["run_best_final_objective"] = best_final
    raw["objective_gap_to_run_best_final"] = np.maximum(
        raw["objective"] - best_final, 1e-14
    )
    final["run_best_final_objective"] = best_final
    final["objective_gap_to_run_best_final"] = np.maximum(
        final["objective"] - best_final, 1e-14
    )
    runtime = time.perf_counter() - started
    raw["run_runtime_seconds"] = runtime
    final["run_runtime_seconds"] = runtime
    return raw, final


def prepare_curve(
    group: pd.DataFrame, x_column: str, y_column: str
) -> tuple[np.ndarray, np.ndarray]:
    values = (
        group[[x_column, y_column]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(x_column)
        .groupby(x_column, as_index=False)
        .last()
    )
    return (
        values[x_column].to_numpy(dtype=float),
        values[y_column].to_numpy(dtype=float),
    )


def average_curve(
    raw: pd.DataFrame,
    x_column: str,
    y_column: str,
    grid_size: int,
    include_zero: bool,
    log_grid: bool,
) -> pd.DataFrame:
    rows = []
    for method_key in core.COMPARISON_METHODS:
        curves = []
        method_rows = raw[raw["method_key"] == method_key]
        for _, group in method_rows.groupby("mc_run"):
            x_values, y_values = prepare_curve(group, x_column, y_column)
            if x_values.size >= 2:
                curves.append((x_values, y_values))
        if not curves:
            continue

        x_end = min(float(x_values.max()) for x_values, _ in curves)
        if include_zero:
            x_start = 0.0
        else:
            starts = [
                float(x_values[x_values > 0.0].min())
                for x_values, _ in curves
                if np.any(x_values > 0.0)
            ]
            if not starts:
                continue
            x_start = max(starts)
        if not x_end > x_start:
            continue

        if log_grid and x_start > 0.0:
            grid = np.geomspace(x_start, x_end, int(grid_size))
        else:
            grid = np.linspace(x_start, x_end, int(grid_size))
        samples = np.vstack(
            [np.interp(grid, x_values, y_values) for x_values, y_values in curves]
        )
        num_runs = samples.shape[0]
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=1) if num_runs > 1 else np.zeros_like(mean)
        sem = std / math.sqrt(num_runs)
        q25 = np.quantile(samples, 0.25, axis=0)
        q75 = np.quantile(samples, 0.75, axis=0)

        for index, x_value in enumerate(grid):
            rows.append(
                {
                    "method_key": method_key,
                    "method": core.METHOD_LABELS[method_key],
                    x_column: float(x_value),
                    f"{y_column}_mean": float(mean[index]),
                    f"{y_column}_std": float(std[index]),
                    f"{y_column}_sem": float(sem[index]),
                    f"{y_column}_q25": float(q25[index]),
                    f"{y_column}_q75": float(q75[index]),
                    f"{y_column}_ci95_low": float(mean[index] - 1.96 * sem[index]),
                    f"{y_column}_ci95_high": float(mean[index] + 1.96 * sem[index]),
                    "num_runs": int(num_runs),
                }
            )
    return pd.DataFrame(rows)


def summarize_final(final: pd.DataFrame, metric_columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for method_key, group in final.groupby("method_key", observed=True):
        row = {
            "method_key": method_key,
            "method": str(group["method"].iloc[0]),
            "num_runs": int(group["mc_run"].nunique()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = std / math.sqrt(values.size)
        rows.append(row)
    return pd.DataFrame(rows)


def write_per_method_tikz_csvs(
    output_dir: Path,
    curve_name: str,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> None:
    tikz_dir = output_dir / "tikz_csv"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    for method_key, group in frame.groupby("method_key", observed=True):
        output = group[
            [
                x_column,
                f"{y_column}_mean",
                f"{y_column}_std",
                f"{y_column}_sem",
                "num_runs",
            ]
        ].rename(
            columns={
                x_column: "x",
                f"{y_column}_mean": "mean",
                f"{y_column}_std": "std",
                f"{y_column}_sem": "sem",
            }
        )
        output.to_csv(tikz_dir / f"{curve_name}_{method_key}.csv", index=False)


def aggregate_outputs(
    output_dir: Path,
    metric_columns: Iterable[str] = BASE_METRICS,
    grid_size: int = 401,
    include_zero: bool = False,
) -> AggregatedOutputs | None:
    raw_files = sorted(output_dir.glob("raw_histories_run_*.csv"))
    final_files = sorted(output_dir.glob("final_run_*.csv"))
    if not raw_files or not final_files:
        return None

    raw = pd.concat((pd.read_csv(path) for path in raw_files), ignore_index=True)
    final = pd.concat((pd.read_csv(path) for path in final_files), ignore_index=True)
    final_summary = summarize_final(final, metric_columns)
    objective_vs_matvec = average_curve(
        raw,
        "matvec_pass_equivalent",
        "objective",
        grid_size,
        include_zero,
        log_grid=True,
    )
    objective_vs_time = average_curve(
        raw,
        "optimization_time_seconds",
        "objective",
        grid_size,
        include_zero,
        log_grid=True,
    )
    feasibility_vs_matvec = average_curve(
        raw,
        "matvec_pass_equivalent",
        "marginal_relative_l1_residual",
        grid_size,
        include_zero,
        log_grid=True,
    )

    raw.to_csv(output_dir / "monte_carlo_raw_histories.csv", index=False)
    final.to_csv(output_dir / "monte_carlo_final_by_run.csv", index=False)
    final_summary.to_csv(
        output_dir / "monte_carlo_final_summary_by_method.csv", index=False
    )
    objective_vs_matvec.to_csv(
        output_dir / "average_objective_vs_matvec.csv", index=False
    )
    objective_vs_time.to_csv(
        output_dir / "average_objective_vs_optimization_time.csv", index=False
    )
    feasibility_vs_matvec.to_csv(
        output_dir / "average_feasibility_vs_matvec.csv", index=False
    )
    write_per_method_tikz_csvs(
        output_dir,
        "objective_vs_matvec",
        objective_vs_matvec,
        "matvec_pass_equivalent",
        "objective",
    )
    write_per_method_tikz_csvs(
        output_dir,
        "objective_vs_optimization_time",
        objective_vs_time,
        "optimization_time_seconds",
        "objective",
    )
    write_per_method_tikz_csvs(
        output_dir,
        "feasibility_vs_matvec",
        feasibility_vs_matvec,
        "matvec_pass_equivalent",
        "marginal_relative_l1_residual",
    )
    return AggregatedOutputs(
        raw=raw,
        final=final,
        final_summary=final_summary,
        objective_vs_matvec=objective_vs_matvec,
        objective_vs_time=objective_vs_time,
        feasibility_vs_matvec=feasibility_vs_matvec,
    )


def paper_curve(
    averaged: pd.DataFrame,
    x_column: str,
    metric: str,
    output_metric: str,
) -> pd.DataFrame:
    columns = {
        f"{metric}_mean": output_metric,
        f"{metric}_ci95_low": f"{output_metric}_lo",
        f"{metric}_ci95_high": f"{output_metric}_hi",
    }
    result = averaged[["method_key", x_column, *columns]].rename(columns=columns)
    result.insert(0, "method", result.pop("method_key").map(PAPER_METHOD_LABELS))
    if result["method"].isna().any():
        raise ValueError("A method is missing from PAPER_METHOD_LABELS.")
    return result


def publish_paper_outputs(paper_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paper_dir.glob("*.csv"):
        shutil.copy2(source, destination / source.name)


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(jsonable(metadata), indent=2) + "\n")
