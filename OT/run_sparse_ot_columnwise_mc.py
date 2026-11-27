#!/usr/bin/env python3
"""Monte Carlo runner for sparse_ot_bcdc_columnwise_entropy_smoothness.ipynb.

The script reuses the notebook definitions directly, varies the problem and
solver seeds across Monte Carlo runs, and writes raw/final/averaged CSV files
for PGFPlots/TikZ.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = SCRIPT_DIR / "sparse_ot_bcdc_columnwise_entropy_smoothness.ipynb"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "sparse_ot_columnwise_mc10"
DEFINITION_CELLS = (2, 4, 6, 8, 10)
METRIC_COLUMNS = (
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


WRIGHTOMEGA_FALLBACK_DEF = r'''
def wrightomega(x):
    """Positive real Wright omega fallback solving w + log(w) = x.

    This replaces scipy.special.wrightomega for the real-valued arrays used by
    the notebook's closed-form entropy updates.
    """
    scalar_input = np.isscalar(x)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    nan_mask = np.isnan(x)
    pos_inf_mask = np.isposinf(x)
    neg_inf_mask = np.isneginf(x)
    finite_mask = np.isfinite(x)
    out[nan_mask] = np.nan
    out[pos_inf_mask] = np.inf
    out[neg_inf_mask] = 0.0

    if np.any(finite_mask):
        xf = x[finite_mask]
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            w = np.where(
                xf > 1.0,
                xf - np.log(np.maximum(xf, tiny)),
                np.exp(xf),
            )
        w = np.maximum(w, tiny)
        for _ in range(30):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                f = w + np.log(w) - xf
                fp = 1.0 + 1.0 / w
                step = f / fp
            step = np.where(np.isfinite(step), step, 0.0)
            w_next = w - step
            bad = (~np.isfinite(w_next)) | (w_next <= 0.0)
            if np.any(bad):
                w_next[bad] = 0.5 * w[bad]
            w_next = np.maximum(w_next, tiny)
            if np.all(np.abs(w_next - w) <= 1e-13 * np.maximum(1.0, w_next)):
                w = w_next
                break
            w = w_next
        out[finite_mask] = w

    if scalar_input:
        return float(out)
    return out
'''

WRIGHTOMEGA_IMPORT_OR_FALLBACK = (
    "try:\n"
    "    from scipy.special import wrightomega\n"
    "except Exception:\n"
    + "\n".join(
        f"    {line}" if line else line
        for line in WRIGHTOMEGA_FALLBACK_DEF.splitlines()
    )
    + "\n"
)


def sanitize_notebook_source(source: str, cell_index: int) -> str:
    if cell_index != 2:
        return source
    source = source.replace("import matplotlib.pyplot as plt\n", "")
    source = source.replace(
        "from scipy.special import wrightomega\n",
        WRIGHTOMEGA_IMPORT_OR_FALLBACK,
    )
    source = re.sub(
        r"\nplt\.rcParams\.update\(\{.*?\n\}\)\n",
        "\n",
        source,
        flags=re.S,
    )
    return source


def load_notebook_namespace(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text())
    module_name = "sparse_ot_columnwise_notebook_defs"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    namespace = module.__dict__
    for cell_index in DEFINITION_CELLS:
        source = "".join(notebook["cells"][cell_index].get("source", []))
        source = sanitize_notebook_source(source, cell_index)
        code = compile(source, f"{notebook_path}#cell{cell_index}", "exec")
        exec(code, namespace)
    return namespace


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def install_lightweight_history_row(namespace: dict) -> None:
    """Drop diagnostic-only full gap/KKT work from recorded history rows."""

    def make_light_history_row(
        problem,
        plan,
        source_marginal,
        target_marginal,
        config,
        iteration,
        sweep,
        counters,
        selected_score,
        method_key=None,
        outer_iteration=None,
        inner_iterations=None,
        subproblem_obj_change=np.nan,
        subproblem_converged=False,
    ):
        method_key = config.selection_rule if method_key is None else method_key
        dense_nnz = namespace["dense_problem_nnz"](problem)
        source_residual = float(
            np.sum(np.abs(source_marginal - problem.source_mass))
            / np.sum(problem.source_mass)
        )
        target_residual = float(
            np.sum(np.abs(target_marginal - problem.target_mass))
            / np.sum(problem.target_mass)
        )
        matvec_pass_equivalent = (
            float(counters["touched_nonzeros"] / dense_nnz)
            if dense_nnz > 0
            else np.nan
        )
        return {
            "method": namespace["METHOD_LABELS"][method_key],
            "method_key": method_key,
            "configured_outer_iterations": int(config.num_sweeps),
            "configured_max_inner_iterations": int(config.max_inner_iterations),
            "outer_iteration": int(
                iteration if outer_iteration is None else outer_iteration
            ),
            "inner_iterations": int(
                iteration if inner_iterations is None else inner_iterations
            ),
            "iteration": int(iteration),
            "sweep": float(sweep),
            "optimization_time_seconds": float(counters["optimization_time"]),
            "wall_clock_time": float(counters["wall_clock_time"]),
            "touched_nonzeros": int(counters["touched_nonzeros"]),
            "column_accesses": int(counters["column_accesses"]),
            "objective_evaluations": int(counters["objective_evaluations"]),
            "matvec_pass_equivalent": matvec_pass_equivalent,
            "matvec_sweeps": matvec_pass_equivalent,
            "selected_score": float(selected_score),
            "subproblem_obj_change": float(subproblem_obj_change),
            "subproblem_converged": bool(subproblem_converged),
            "objective": namespace["objective_value"](
                problem, plan, source_marginal, target_marginal
            ),
            "source_marginal_relative_l1_residual": source_residual,
            "target_marginal_relative_l1_residual": target_residual,
            "marginal_relative_l1_residual": max(source_residual, target_residual),
            "mean_step_size": float(counters["mean_step_size"]),
            "acceptance_rate": float(counters["acceptance_rate"]),
            "number_backtracking_steps": int(
                counters["number_backtracking_steps"]
            ),
        }

    namespace["make_history_row"] = make_light_history_row


def build_problem(namespace: dict, config: dict):
    return namespace["make_clustered_problem"](
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


def build_base_solver_config(namespace: dict, config: dict, problem):
    SolverConfig = namespace["SolverConfig"]
    return SolverConfig(
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


def make_run_config(base_config: dict, run_index: int) -> dict:
    config = copy.deepcopy(base_config)
    config["problem_seed"] = int(base_config["problem_seed"]) + run_index
    config["solver_seed"] = int(base_config["solver_seed"]) + 1009 * run_index
    return config


def run_one(namespace: dict, run_index: int, base_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = make_run_config(base_config, run_index)
    problem = build_problem(namespace, config)
    initial_plan = namespace["initialize_plan"](problem)
    base_solver_config = build_base_solver_config(namespace, config, problem)

    outer_iterations_by_method = namespace["resolve_method_integer_budgets"](
        config.get("outer_iterations_by_method"),
        default=config["num_sweeps"],
        name="outer_iterations_by_method",
    )
    max_inner_iterations_by_method = namespace["resolve_method_integer_budgets"](
        config.get("max_inner_iterations_by_method"),
        default=config["max_inner_iterations"],
        name="max_inner_iterations_by_method",
    )

    histories = []
    started = time.perf_counter()
    for method_key in namespace["COMPARISON_METHODS"]:
        method_started = time.perf_counter()
        cfg = namespace["build_method_config"](
            base_solver_config,
            method_key,
            outer_iterations_by_method,
            max_inner_iterations_by_method,
        )
        cfg.validate(problem.num_target)

        if method_key in namespace["BCDC_COMPARISON_RULES"]:
            _, history = namespace["solve_bcdc"](problem, cfg, initial_plan)
        elif method_key == "full_entropy":
            _, history = namespace["solve_full_dca"](
                problem, cfg, initial_plan, geometry="entropy"
            )
        elif method_key == "full_euclidean":
            _, history = namespace["solve_full_dca"](
                problem, cfg, initial_plan, geometry="euclidean"
            )
        else:
            raise ValueError(method_key)

        history = history.copy()
        history["mc_run"] = int(run_index)
        history["problem_seed"] = int(config["problem_seed"])
        history["solver_seed"] = int(config["solver_seed"])
        history["method_runtime_seconds"] = time.perf_counter() - method_started
        histories.append(history)

        last = history.iloc[-1]
        print(
            f"[run {run_index:02d}] {namespace['METHOD_LABELS'][method_key]:42s} "
            f"obj={last['objective']:.6g} "
            f"feas={last['marginal_relative_l1_residual']:.3e} "
            f"matvec={last['matvec_pass_equivalent']:.1f} "
            f"opt_time={last['optimization_time_seconds']:.2f}s",
            flush=True,
        )

    raw = pd.concat(histories, ignore_index=True)
    final = (
        raw.sort_values(["mc_run", "method_key", "iteration"])
        .groupby(["mc_run", "method_key"], as_index=False, observed=True)
        .tail(1)
        .copy()
    )
    best_final_by_run = final.groupby("mc_run")["objective"].min().rename("run_best_final_objective")
    raw = raw.merge(best_final_by_run, on="mc_run", how="left")
    raw["objective_gap_to_run_best_final"] = np.maximum(
        raw["objective"] - raw["run_best_final_objective"], 1e-14
    )
    final = final.merge(best_final_by_run, on="mc_run", how="left")
    final["objective_gap_to_run_best_final"] = np.maximum(
        final["objective"] - final["run_best_final_objective"], 1e-14
    )
    raw["run_runtime_seconds"] = time.perf_counter() - started
    final["run_runtime_seconds"] = time.perf_counter() - started
    return raw, final


def prepare_curve(group: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    sub = group[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub.sort_values(x_col).groupby(x_col, as_index=False).last()
    return (
        sub[x_col].to_numpy(dtype=float),
        sub[y_col].to_numpy(dtype=float),
    )


def average_curve(
    raw: pd.DataFrame,
    method_order: tuple[str, ...],
    method_labels: dict,
    x_col: str,
    y_col: str,
    grid_size: int,
    include_zero: bool,
) -> pd.DataFrame:
    rows = []
    for method_key in method_order:
        curves = []
        for _, group in raw[raw["method_key"] == method_key].groupby("mc_run"):
            x_values, y_values = prepare_curve(group, x_col, y_col)
            if x_values.size < 2:
                continue
            curves.append((x_values, y_values))
        if not curves:
            continue

        ends = [float(x_values.max()) for x_values, _ in curves]
        x_end = min(ends)
        if include_zero:
            x_start = 0.0
        else:
            positive_starts = [
                float(x_values[x_values > 0.0].min())
                for x_values, _ in curves
                if np.any(x_values > 0.0)
            ]
            if not positive_starts:
                continue
            x_start = max(positive_starts)
        if not x_end > x_start:
            continue

        grid = np.linspace(x_start, x_end, int(grid_size))
        values = np.vstack([
            np.interp(grid, x_values, y_values)
            for x_values, y_values in curves
        ])
        n_runs = values.shape[0]
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean)
        sem = std / math.sqrt(n_runs)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)

        for i, x_value in enumerate(grid):
            rows.append({
                "method_key": method_key,
                "method": method_labels[method_key],
                x_col: float(x_value),
                f"{y_col}_mean": float(mean[i]),
                f"{y_col}_std": float(std[i]),
                f"{y_col}_sem": float(sem[i]),
                f"{y_col}_q25": float(q25[i]),
                f"{y_col}_q75": float(q75[i]),
                f"{y_col}_ci95_low": float(mean[i] - 1.96 * sem[i]),
                f"{y_col}_ci95_high": float(mean[i] + 1.96 * sem[i]),
                "num_runs": int(n_runs),
            })
    return pd.DataFrame(rows)


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_key, group in final.groupby("method_key", observed=True):
        row = {
            "method_key": method_key,
            "method": str(group["method"].iloc[0]),
            "num_runs": int(group["mc_run"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            row[f"{metric}_sem"] = (
                row[f"{metric}_std"] / math.sqrt(values.size)
                if values.size > 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_per_method_tikz_csvs(
    output_dir: Path,
    curve_name: str,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    tikz_dir = output_dir / "tikz_csv"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    mean_col = f"{y_col}_mean"
    std_col = f"{y_col}_std"
    sem_col = f"{y_col}_sem"
    for method_key, group in df.groupby("method_key", observed=True):
        out = group[[x_col, mean_col, std_col, sem_col, "num_runs"]].copy()
        out = out.rename(columns={
            x_col: "x",
            mean_col: "mean",
            std_col: "std",
            sem_col: "sem",
        })
        out.to_csv(tikz_dir / f"{curve_name}_{method_key}.csv", index=False)


def aggregate_outputs(
    output_dir: Path,
    method_order: tuple[str, ...],
    method_labels: dict,
    grid_size: int,
    include_zero: bool,
) -> None:
    raw_files = sorted(output_dir.glob("raw_histories_run_*.csv"))
    final_files = sorted(output_dir.glob("final_run_*.csv"))
    if not raw_files or not final_files:
        return

    raw = pd.concat((pd.read_csv(path) for path in raw_files), ignore_index=True)
    final = pd.concat((pd.read_csv(path) for path in final_files), ignore_index=True)
    raw.to_csv(output_dir / "monte_carlo_raw_histories.csv", index=False)
    final.to_csv(output_dir / "monte_carlo_final_by_run.csv", index=False)

    final_summary = summarize_final(final)
    final_summary.to_csv(output_dir / "monte_carlo_final_summary_by_method.csv", index=False)

    objective_vs_matvec = average_curve(
        raw,
        method_order=method_order,
        method_labels=method_labels,
        x_col="matvec_pass_equivalent",
        y_col="objective",
        grid_size=grid_size,
        include_zero=include_zero,
    )
    objective_vs_time = average_curve(
        raw,
        method_order=method_order,
        method_labels=method_labels,
        x_col="optimization_time_seconds",
        y_col="objective",
        grid_size=grid_size,
        include_zero=include_zero,
    )
    feasibility_vs_matvec = average_curve(
        raw,
        method_order=method_order,
        method_labels=method_labels,
        x_col="matvec_pass_equivalent",
        y_col="marginal_relative_l1_residual",
        grid_size=grid_size,
        include_zero=include_zero,
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument(
        "--include-zero-grid",
        action="store_true",
        help=(
            "Include x=0 in averaged curve CSVs. By default the averaged grids "
            "start at the first common positive x value, which is safer for "
            "log-scale PGFPlots axes. Raw histories always include x=0."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Monte Carlo runs whose per-run CSV files already exist.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip the notebook's lightweight implementation checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    namespace = load_notebook_namespace(args.notebook)
    base_config = copy.deepcopy(namespace["CONFIG"])

    metadata = {
        "notebook": str(args.notebook),
        "num_runs_requested": int(args.num_runs),
        "grid_size": int(args.grid_size),
        "include_zero_grid": bool(args.include_zero_grid),
        "definition_cells": list(DEFINITION_CELLS),
        "base_config": jsonable(base_config),
        "history_note": (
            "Optimizer definitions are loaded from the notebook. Per-iteration "
            "history rows omit diagnostic-only full gap/KKT scans, but retain "
            "objective, feasibility, matvec, and timing metrics."
        ),
    }
    (args.output_dir / "monte_carlo_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    if not args.skip_checks:
        check_config = make_run_config(base_config, 0)
        check_problem = build_problem(namespace, check_config)
        check_solver_config = build_base_solver_config(
            namespace, check_config, check_problem
        )
        print("Running notebook correctness checks...", flush=True)
        assert namespace["run_correctness_checks"](check_problem, check_solver_config)
        print("Correctness checks passed.", flush=True)

    install_lightweight_history_row(namespace)

    method_budget_table = namespace["make_method_budget_table"](
        namespace["resolve_method_integer_budgets"](
            base_config.get("outer_iterations_by_method"),
            default=base_config["num_sweeps"],
            name="outer_iterations_by_method",
        ),
        namespace["resolve_method_integer_budgets"](
            base_config.get("max_inner_iterations_by_method"),
            default=base_config["max_inner_iterations"],
            name="max_inner_iterations_by_method",
        ),
        int(base_config["num_target"]),
    )
    method_budget_table.to_csv(args.output_dir / "method_budget_table.csv", index=False)

    total_started = time.perf_counter()
    for run_index in range(args.num_runs):
        raw_path = args.output_dir / f"raw_histories_run_{run_index:03d}.csv"
        final_path = args.output_dir / f"final_run_{run_index:03d}.csv"
        if args.resume and raw_path.exists() and final_path.exists():
            print(f"[run {run_index:02d}] already exists; skipping.", flush=True)
            continue

        print(f"[run {run_index:02d}] starting", flush=True)
        raw, final = run_one(namespace, run_index, base_config)
        raw.to_csv(raw_path, index=False)
        final.to_csv(final_path, index=False)
        aggregate_outputs(
            args.output_dir,
            method_order=tuple(namespace["COMPARISON_METHODS"]),
            method_labels=namespace["METHOD_LABELS"],
            grid_size=args.grid_size,
            include_zero=args.include_zero_grid,
        )
        elapsed = time.perf_counter() - total_started
        print(
            f"[run {run_index:02d}] saved. Total elapsed {elapsed / 60.0:.2f} min.",
            flush=True,
        )

    aggregate_outputs(
        args.output_dir,
        method_order=tuple(namespace["COMPARISON_METHODS"]),
        method_labels=namespace["METHOD_LABELS"],
        grid_size=args.grid_size,
        include_zero=args.include_zero_grid,
    )
    print(f"Done. CSV outputs are in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
