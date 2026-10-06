"""Sparse unbalanced-OT optimization methods shared by all experiments.

It contains the problem model, update rules, solvers, and work accounting,
but no data loading, experiment orchestration, plotting, or file I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import gammaln, wrightomega

EPS = 1e-12


@dataclass(frozen=True)
class SparseOTProblem:
    cost: np.ndarray
    source_mass: np.ndarray
    target_mass: np.ndarray
    source_kl_weight: float = 20.0
    target_kl_weight: float = 20.0
    quadratic_weight: float = 1.0
    sparsity_weight: float = 1.0
    top_k: int = 2

    @property
    def num_source(self):
        return int(self.cost.shape[0])

    @property
    def num_target(self):
        return int(self.cost.shape[1])

    @property
    def entropy_relative_smoothness(self):
        return float(self.source_kl_weight + self.target_kl_weight)


@dataclass(frozen=True)
class SolverConfig:
    num_sweeps: int = 100
    selection_rule: str = "uniform"
    candidate_batch_size: int = 32
    seed: int = 0
    record_every_sweeps: int = 5
    min_plan_value: float = 1e-300
    relative_smoothness_scale: float = 1.0
    block_log_radius: float = 4.0
    sampling: str = "random_reshuffling"
    max_inner_iterations: int = 50
    inner_tol: float = 1e-12
    # "relative_change": stop an inner solve when the subproblem objective
    # changes by at most inner_tol (relative).  "certificate": stop when the
    # certified bound on the subproblem suboptimality is at most
    # eps_k = inner_rho * max(Delta_k, decrease so far); see
    # subproblem_certificate.
    inner_stopping: str = "relative_change"
    inner_rho: float = 0.1
    # Record Gamma_k = n max_j Delta_j / sum_j Delta_j (and its random-batch
    # counterpart) at every history row.
    log_gamma: bool = False

    def validate(self, num_target):
        valid = {"uniform", "gradient", "lipschitz", "bregman_gap"}
        if self.selection_rule not in valid:
            raise ValueError(f"selection_rule must be one of {sorted(valid)}")
        if self.num_sweeps <= 0 or self.record_every_sweeps <= 0:
            raise ValueError("Sweep counts must be positive.")
        if not 1 <= self.candidate_batch_size <= num_target:
            raise ValueError("candidate_batch_size must be between 1 and num_target.")
        if self.min_plan_value <= 0.0:
            raise ValueError("min_plan_value must be strictly positive.")
        if self.relative_smoothness_scale <= 0.0:
            raise ValueError("relative_smoothness_scale must be positive.")
        if self.block_log_radius is not None:
            radius = float(self.block_log_radius)
            if np.isnan(radius) or radius <= 0.0:
                raise ValueError("block_log_radius must be positive or np.inf.")
            if np.isfinite(radius) and radius > 50.0:
                raise ValueError("Use np.inf to recover the global entropy constant.")
        if self.sampling not in {"uniform", "random_reshuffling"}:
            raise ValueError("sampling must be 'uniform' or 'random_reshuffling'.")
        if self.max_inner_iterations <= 0:
            raise ValueError("max_inner_iterations must be positive.")
        if self.inner_tol < 0.0:
            raise ValueError("inner_tol must be nonnegative.")
        if self.inner_stopping not in {"relative_change", "certificate"}:
            raise ValueError("inner_stopping must be 'relative_change' or 'certificate'.")
        if not 0.0 <= self.inner_rho < 1.0:
            raise ValueError("inner_rho must be in [0, 1).")


CONFIG = {
    "num_source": 256,
    "num_target": 512,
    "dimension": 128,
    "top_k": 2,
    "source_kl_weight": 100.0,
    "target_kl_weight": 100.0,
    "quadratic_weight": 0.5,
    "sparsity_weight": 10.0,
    "target_mass_log_std": 0.0,
    "outlier_fraction": 0.0,
    "outlier_shift": 0.0,
    "noise_std": 0.0,
    "problem_seed": 0,
    "solver_seed": 0,
    # Fallback outer-iteration budget.  Per-method values below override it.
    "num_sweeps": 100,
    # For BCDC methods, these are sweeps, and one sweep equals num_target
    # one-column updates.  For full DCA methods, these are outer DCA
    # iterations, each freezing the full top-Q subgradient once.
    "outer_iterations_by_method": {
        "lipschitz": 1000,
        "uniform": 6000,
        "bregman_gap": 1000,
        "full_entropy": 100,
        "full_euclidean": 100,
    },
    "record_every_sweeps": 1,
    "min_plan_value": 1e-300,
    "relative_smoothness_scale": 1.0,
    # Paper's Delta in the column trust region |log(z_i / p_ij)| <= Delta.
    # Use np.inf to recover the previous global L = tau_a + tau_b update.
    "block_log_radius": 4.0,
    "candidate_batch_size": 8,  # GS-only parameter
    # Fallback full-DCA inner budget.  Per-method values below override it.
    "max_inner_iterations": 100,
    "max_inner_iterations_by_method": {
        "full_entropy": 100,
        "full_euclidean": 100,
    },
    "inner_tol": 1e-9,
    "inner_stopping": "relative_change",
    "inner_rho": 0.1,
    "log_gamma": False,
}


def generalized_kl(x, y):
    x = np.maximum(np.asarray(x, dtype=np.float64), EPS)
    y = np.maximum(np.asarray(y, dtype=np.float64), EPS)
    return float(np.sum(x * np.log(x / y) - x + y))


def make_clustered_problem(
    num_source=64,
    num_target=120,
    dimension=4,
    top_k=2,
    source_kl_weight=20.0,
    target_kl_weight=20.0,
    quadratic_weight=1.0,
    sparsity_weight=1.0,
    target_mass_log_std=0.4,
    outlier_fraction=0.1,
    outlier_shift=3.0,
    noise_std=0.05,
    seed=0,
):
    if min(num_source, num_target, dimension) <= 0:
        raise ValueError("Problem dimensions must be positive.")
    if not 1 <= top_k <= num_source:
        raise ValueError("top_k must be between 1 and num_source.")

    rng = np.random.default_rng(seed)
    source_points = rng.normal(size=(num_source, dimension))
    source_points /= np.maximum(
        np.linalg.norm(source_points, axis=1, keepdims=True), EPS
    )

    support = np.vstack(
        [rng.choice(num_source, size=top_k, replace=False) for _ in range(num_target)]
    )
    weights = rng.random((num_target, top_k))
    weights /= weights.sum(axis=1, keepdims=True)
    target_points = np.zeros((num_target, dimension))
    for q in range(top_k):
        target_points += weights[:, q, None] * source_points[support[:, q]]
    target_points += noise_std * rng.normal(size=target_points.shape)

    num_outliers = int(round(outlier_fraction * num_target))
    if num_outliers:
        idx = rng.choice(num_target, size=num_outliers, replace=False)
        directions = rng.normal(size=(num_outliers, dimension))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), EPS)
        target_points[idx] += outlier_shift * directions

    diff = source_points[:, None, :] - target_points[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    target_mass = np.exp(target_mass_log_std * rng.normal(size=num_target))
    target_mass *= num_target / target_mass.sum()
    source_mass = np.full(num_source, target_mass.sum() / num_source)
    return SparseOTProblem(
        cost=np.ascontiguousarray(cost),
        source_mass=np.ascontiguousarray(source_mass),
        target_mass=np.ascontiguousarray(target_mass),
        source_kl_weight=float(source_kl_weight),
        target_kl_weight=float(target_kl_weight),
        quadratic_weight=float(quadratic_weight),
        sparsity_weight=float(sparsity_weight),
        top_k=int(top_k),
    )


def initialize_plan(problem):
    total = float(problem.source_mass.sum())
    plan = np.outer(problem.source_mass, problem.target_mass) / total
    return np.maximum(plan, EPS)


def topq_indices(x, q):
    q = min(max(int(q), 0), x.size)
    if q == 0:
        return np.empty(0, dtype=np.int64)
    if q == x.size:
        return np.arange(x.size, dtype=np.int64)
    # One admissible extreme subgradient is enough, including at ties.
    return np.argpartition(x, -q)[-q:]


def topq_value(x, q):
    return float(np.sum(x[topq_indices(x, q)]))


def selected_topq_subgradient(x, q, weight):
    v = np.zeros_like(x, dtype=np.float64)
    v[topq_indices(x, q)] = weight
    return v


def objective_value(problem, plan, source_marginal=None, target_marginal=None):
    if source_marginal is None:
        source_marginal = plan.sum(axis=1)
    if target_marginal is None:
        target_marginal = plan.sum(axis=0)
    trimmed = sum(
        float(plan[:, j].sum()) - topq_value(plan[:, j], problem.top_k)
        for j in range(problem.num_target)
    )
    return float(
        np.sum(problem.cost * plan)
        + problem.source_kl_weight
        * generalized_kl(source_marginal, problem.source_mass)
        + problem.target_kl_weight
        * generalized_kl(target_marginal, problem.target_mass)
        + 0.5 * problem.quadratic_weight * np.sum(plan * plan)
        + problem.sparsity_weight * trimmed
    )


def block_gradient_f(problem, source_marginal, target_marginal, j):
    r = np.maximum(source_marginal, EPS)
    sj = max(float(target_marginal[j]), EPS)
    return (
        problem.cost[:, j]
        + problem.source_kl_weight * np.log(r / problem.source_mass)
        + problem.target_kl_weight * np.log(sj / problem.target_mass[j])
    )


def block_gradient_f_batch(problem, source_marginal, target_marginal, candidates):
    candidates = np.asarray(candidates, dtype=np.int64)
    r = np.maximum(source_marginal, EPS)
    sj = np.maximum(target_marginal[candidates], EPS)
    return (
        problem.cost[:, candidates]
        + problem.source_kl_weight * np.log(r / problem.source_mass)[:, None]
        + problem.target_kl_weight
        * np.log(sj / problem.target_mass[candidates])[None, :]
    )


def log_trust_region_bounds(p, log_radius, min_value=EPS):
    if log_radius is None:
        return None, None
    radius = float(log_radius)
    if not np.isfinite(radius):
        return None, None
    p = np.maximum(np.asarray(p, dtype=np.float64), min_value)
    lower = np.maximum(p * np.exp(-radius), min_value)
    upper = np.maximum(p * np.exp(radius), lower)
    return lower, upper


def entropy_bcdc_candidate(
    p, d, relative_smoothness, gamma, min_value=EPS, log_radius=None
):
    p = np.maximum(np.asarray(p, dtype=np.float64), min_value)
    d = np.asarray(d, dtype=np.float64)
    L = np.asarray(relative_smoothness, dtype=np.float64)
    gamma = float(gamma)
    if np.any(L <= 0.0):
        raise ValueError("The entropy relative-smoothness weight must be positive.")
    if gamma == 0.0:
        log_z = np.log(p) - d / L
        z = np.exp(np.clip(log_z, np.log(min_value), np.log(np.finfo(float).max)))
    else:
        log_argument = np.log(gamma * p / L) - d / L
        z = (L / gamma) * wrightomega(log_argument)
    lower, upper = log_trust_region_bounds(p, log_radius, min_value)
    if lower is not None:
        z = np.minimum(np.maximum(z, lower), upper)
    return np.maximum(np.asarray(z, dtype=np.float64), min_value)


def columnwise_entropy_smoothness(problem, source_marginal, p, config):
    radius = config.block_log_radius
    base = problem.entropy_relative_smoothness
    if radius is None or not np.isfinite(float(radius)):
        return float(base * config.relative_smoothness_scale)

    alpha = np.exp(float(radius))
    r = np.maximum(np.asarray(source_marginal, dtype=np.float64), EPS)
    p = np.maximum(np.asarray(p, dtype=np.float64), config.min_plan_value)
    denominator = np.maximum(r + np.expm1(float(radius)) * p, EPS)
    source_fraction = np.clip(alpha * p / denominator, 0.0, 1.0)
    local_L = problem.target_kl_weight + problem.source_kl_weight * float(
        np.max(source_fraction)
    )
    return float(local_L * config.relative_smoothness_scale)


def columnwise_entropy_smoothness_batch(problem, source_marginal, p, config):
    radius = config.block_log_radius
    p = np.asarray(p, dtype=np.float64)
    if radius is None or not np.isfinite(float(radius)):
        return np.full(
            p.shape[1],
            problem.entropy_relative_smoothness * config.relative_smoothness_scale,
            dtype=np.float64,
        )

    alpha = np.exp(float(radius))
    r = np.maximum(np.asarray(source_marginal, dtype=np.float64), EPS)[:, None]
    p_safe = np.maximum(p, config.min_plan_value)
    denominator = np.maximum(r + np.expm1(float(radius)) * p_safe, EPS)
    source_fraction = np.clip(alpha * p_safe / denominator, 0.0, 1.0)
    local_L = problem.target_kl_weight + problem.source_kl_weight * np.max(
        source_fraction, axis=0
    )
    return np.asarray(local_L * config.relative_smoothness_scale, dtype=np.float64)


def entropy_kl(z, p):
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
    return float(np.sum(z * np.log(z / p) - z + p))


def entropy_kl_columns(z, p):
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
    return np.sum(z * np.log(z / p) - z + p, axis=0)


def relative_objective_change(previous, current):
    return float(abs(current - previous) / max(1.0, abs(previous)))


def projected_block_gradient(gradient, p, min_value):
    gradient = np.asarray(gradient, dtype=np.float64).copy()
    active_lower = p <= min_value * (1.0 + 1e-8)
    # At the numerical lower bound, a positive gradient only requests an infeasible decrease.
    gradient[active_lower & (gradient > 0.0)] = 0.0
    return gradient


def projected_block_gradient_batch(gradient, p, min_value):
    projected = np.asarray(gradient, dtype=np.float64).copy()
    active_lower = p <= min_value * (1.0 + 1e-8)
    projected[active_lower & (projected > 0.0)] = 0.0
    return projected


def local_euclidean_lipschitz_bound(problem, source_marginal, target_marginal, j):
    r_min = max(float(np.min(source_marginal)), EPS)
    sj = max(float(target_marginal[j]), EPS)
    return float(
        problem.quadratic_weight
        + problem.source_kl_weight / r_min
        + problem.target_kl_weight * problem.num_source / sj
    )


def local_euclidean_lipschitz_bounds(
    problem, source_marginal, target_marginal, candidates
):
    candidates = np.asarray(candidates, dtype=np.int64)
    r_min = max(float(np.min(source_marginal)), EPS)
    sj = np.maximum(target_marginal[candidates], EPS)
    return (
        problem.quadratic_weight
        + problem.source_kl_weight / r_min
        + problem.target_kl_weight * problem.num_source / sj
    )


def selected_topq_subgradient_batch(values, q, weight):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array.")
    m, b = values.shape
    q = min(max(int(q), 0), m)
    v = np.zeros_like(values, dtype=np.float64)
    if q == 0 or b == 0:
        return v
    if q == m:
        v[:, :] = weight
        return v
    rows = np.argpartition(values, -q, axis=0)[-q:, :]
    cols = np.arange(b, dtype=np.int64)[None, :]
    v[rows, cols] = weight
    return v


def block_terms_with_selected_subgradient(
    problem, plan, source_marginal, target_marginal, j, v_j=None
):
    p = plan[:, j]
    grad_f = block_gradient_f(problem, source_marginal, target_marginal, j)
    if v_j is None:
        v_j = selected_topq_subgradient(p, problem.top_k, problem.sparsity_weight)
    else:
        v_j = np.asarray(v_j, dtype=np.float64).copy()
    d = grad_f + problem.sparsity_weight - v_j
    selected_dc_gradient = d + problem.quadratic_weight * p
    return p, v_j, d, selected_dc_gradient


def block_terms_with_selected_subgradient_batch(
    problem, plan, source_marginal, target_marginal, candidates, v_batch=None
):
    candidates = np.asarray(candidates, dtype=np.int64)
    p = plan[:, candidates]
    grad_f = block_gradient_f_batch(
        problem, source_marginal, target_marginal, candidates
    )
    if v_batch is None:
        v_batch = selected_topq_subgradient_batch(
            p, problem.top_k, problem.sparsity_weight
        )
    else:
        v_batch = np.asarray(v_batch, dtype=np.float64).copy()
    d = grad_f + problem.sparsity_weight - v_batch
    selected_dc_gradient = d + problem.quadratic_weight * p
    return p, v_batch, d, selected_dc_gradient


def evaluate_block(
    problem, plan, source_marginal, target_marginal, j, config, need_candidate, v_j=None
):
    p, selected_v, d, selected_dc_gradient = block_terms_with_selected_subgradient(
        problem, plan, source_marginal, target_marginal, j, v_j=v_j
    )
    projected_gradient = projected_block_gradient(
        selected_dc_gradient, p, config.min_plan_value
    )

    candidate = None
    gap = None
    if need_candidate:
        L = columnwise_entropy_smoothness(problem, source_marginal, p, config)
        candidate = entropy_bcdc_candidate(
            p,
            d,
            L,
            problem.quadratic_weight,
            config.min_plan_value,
            log_radius=config.block_log_radius,
        )
        gap = (
            float(np.dot(d, p - candidate))
            + 0.5
            * problem.quadratic_weight
            * float(np.dot(p, p) - np.dot(candidate, candidate))
            - L * entropy_kl(candidate, p)
        )
        gap = max(float(gap), 0.0)

    if config.selection_rule == "uniform":
        score = np.nan
    elif config.selection_rule == "gradient":
        score = float(np.linalg.norm(projected_gradient))
    elif config.selection_rule == "lipschitz":
        ell = local_euclidean_lipschitz_bound(
            problem, source_marginal, target_marginal, j
        )
        score = float(np.linalg.norm(projected_gradient) / np.sqrt(max(ell, EPS)))
    elif config.selection_rule == "bregman_gap":
        score = float(gap)
    else:
        raise ValueError(config.selection_rule)

    return score, candidate, selected_dc_gradient, gap, selected_v


def evaluate_block_batch(
    problem, plan, source_marginal, target_marginal, candidates, config, need_candidate
):
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("At least one candidate column is required.")
    need_candidate = bool(need_candidate or config.selection_rule == "bregman_gap")
    p, selected_v, d, selected_dc_gradient = (
        block_terms_with_selected_subgradient_batch(
            problem, plan, source_marginal, target_marginal, candidates
        )
    )
    projected_gradient = projected_block_gradient_batch(
        selected_dc_gradient, p, config.min_plan_value
    )

    candidate = None
    gap = None
    if need_candidate:
        L = columnwise_entropy_smoothness_batch(problem, source_marginal, p, config)
        candidate = entropy_bcdc_candidate(
            p,
            d,
            L,
            problem.quadratic_weight,
            config.min_plan_value,
            log_radius=config.block_log_radius,
        )
        gap = (
            np.sum(d * (p - candidate), axis=0)
            + 0.5
            * problem.quadratic_weight
            * (np.sum(p * p, axis=0) - np.sum(candidate * candidate, axis=0))
            - L * entropy_kl_columns(candidate, p)
        )
        gap = np.maximum(gap.astype(np.float64), 0.0)

    if config.selection_rule == "uniform":
        score = np.full(candidates.size, np.nan, dtype=np.float64)
    elif config.selection_rule == "gradient":
        score = np.linalg.norm(projected_gradient, axis=0)
    elif config.selection_rule == "lipschitz":
        ell = local_euclidean_lipschitz_bounds(
            problem, source_marginal, target_marginal, candidates
        )
        score = np.linalg.norm(projected_gradient, axis=0) / np.sqrt(
            np.maximum(ell, EPS)
        )
    elif config.selection_rule == "bregman_gap":
        score = gap
    else:
        raise ValueError(config.selection_rule)

    return score, candidate, selected_dc_gradient, gap, selected_v


def solve_selected_coordinate_subproblem(
    problem,
    plan,
    source_marginal,
    target_marginal,
    j,
    selected_v,
    config,
    precomputed_candidate=None,
):
    if precomputed_candidate is not None:
        return precomputed_candidate
    _, candidate, _, _, _ = evaluate_block(
        problem,
        plan,
        source_marginal,
        target_marginal,
        j,
        config,
        need_candidate=True,
        v_j=selected_v,
    )
    return candidate


def column_subproblem_objective(
    problem, plan_col, source_marginal, target_marginal_j, j, v_j
):
    """Frozen (v_j fixed) DCA subproblem objective restricted to column j, used
    only as a relative-change stopping criterion for repeated inner solves."""
    r = np.maximum(source_marginal, EPS)
    sj = max(float(target_marginal_j), EPS)
    return float(
        np.dot(problem.cost[:, j], plan_col)
        + problem.source_kl_weight * generalized_kl(r, problem.source_mass)
        + problem.target_kl_weight
        * generalized_kl(np.array([sj]), np.array([problem.target_mass[j]]))
        + 0.5 * problem.quadratic_weight * float(np.dot(plan_col, plan_col))
        + problem.sparsity_weight * float(np.sum(plan_col))
        - float(np.dot(v_j, plan_col))
    )


def column_gradient_f(problem, source_marginal, target_marginal_j, j):
    """Same as block_gradient_f, but takes the column's scalar target marginal
    directly instead of indexing a full target_marginal array -- lets the
    inner loop re-linearize around an updated column without touching the
    rest of the plan/target_marginal array."""
    r = np.maximum(source_marginal, EPS)
    sj = max(float(target_marginal_j), EPS)
    return (
        problem.cost[:, j]
        + problem.source_kl_weight * np.log(r / problem.source_mass)
        + problem.target_kl_weight * np.log(sj / problem.target_mass[j])
    )


def solve_selected_coordinate_subproblem_inner_loop(
    problem,
    plan,
    source_marginal,
    target_marginal,
    j,
    selected_v,
    config,
    precomputed_candidate=None,
    max_inner=None,
):
    """Repeatedly solve the selected column's entropy-BCDC candidate, up to
    max_inner times (defaults to config.max_inner_iterations, i.e. the flat
    schedule, when not given explicitly), re-linearizing the smooth gradient
    around the current (updated) column value at every inner step, while
    holding the DC subgradient anchor `selected_v` fixed (it is only
    refreshed at the next outer DCA iteration, matching the frozen-subproblem
    convention already used by solve_full_dca / PIP). Mirrors the pattern of
    the full-dimensional inner loop in solve_full_dca, applied per column.
    Operates on the column vector directly (never copies the full plan
    matrix), so it stays as cheap as the existing single-shot update per
    inner step.

    Passing an explicit `max_inner` (e.g. from a per-outer-step schedule such
    as t_k=k, see solve_bcdc's `inner_iterations_schedule`) overrides
    config.max_inner_iterations just for this call, without needing a
    modified config object per outer step.

    Returns (final_plan_column, updated_source_marginal, inner_iterations_used, converged).
    """
    if max_inner is None:
        max_inner = config.max_inner_iterations
    max_inner = int(max_inner)

    p0 = plan[:, j]
    prev_obj = column_subproblem_objective(
        problem, p0, source_marginal, float(target_marginal[j]), j, selected_v
    )

    plan_col = p0
    src = source_marginal
    start_inner = 1
    converged = False
    if precomputed_candidate is not None:
        # `precomputed_candidate` (if any) was already produced by the selection
        # phase's own call to evaluate_block -- i.e. it *is* inner step 1, not a
        # new anchor to step from. Count it as such rather than re-solving it,
        # so max_inner=1 reduces exactly to the old single-shot path.
        src = src + (precomputed_candidate - plan_col)
        plan_col = precomputed_candidate
        tgt_j = float(plan_col.sum())
        curr_obj = column_subproblem_objective(problem, plan_col, src, tgt_j, j, selected_v)
        obj_change = relative_objective_change(prev_obj, curr_obj)
        prev_obj = curr_obj
        if obj_change <= config.inner_tol or max_inner <= 1:
            converged = obj_change <= config.inner_tol
            return plan_col, src, 1, converged
        start_inner = 2

    tgt_j = float(plan_col.sum())
    inner = start_inner - 1
    for inner in range(start_inner, max_inner + 1):
        grad_f = column_gradient_f(problem, src, tgt_j, j)
        d = grad_f + problem.sparsity_weight - selected_v
        L = columnwise_entropy_smoothness(problem, src, plan_col, config)
        candidate = entropy_bcdc_candidate(
            plan_col,
            d,
            L,
            problem.quadratic_weight,
            config.min_plan_value,
            log_radius=config.block_log_radius,
        )

        src = src + (candidate - plan_col)
        plan_col = candidate
        tgt_j = float(plan_col.sum())

        curr_obj = column_subproblem_objective(problem, plan_col, src, tgt_j, j, selected_v)
        obj_change = relative_objective_change(prev_obj, curr_obj)
        prev_obj = curr_obj
        if obj_change <= config.inner_tol:
            converged = True
            break

    return plan_col, src, int(inner), converged



def subproblem_certificate(gradient, z, rest, tau, gamma, bisection_steps=60):
    """Upper bound on phi_hat(z) - min_{y >= 0} phi_hat(y) at a feasible z.

    phi_hat = H + R with H(y) = tau sum_i KL(rest_i + sum_k y_ik | a_i)
    + gamma/2 ||y||^2 + (linear terms) and R the target-marginal KL, which is
    convex.  For the minimizer y*,
        phi_hat(z) - phi_hat(y*) <= <g, z - y*> - D_H(y*, z) - D_R(y*, z),
    so dropping D_R >= 0 and maximizing over all y >= 0 bounds the left-hand
    side; this is the Bregman gap of the subproblem in the geometry of H.

    z has shape (m, k): one column (k = 1, rest = row sums of the other
    columns) or the full plan (k = n, rest = 0).  Writing the row KL through
    its conjugate, tau D(s, w) = sup_mu [mu s - tau w (exp(mu/tau) - 1)], gives
    for every mu the closed-form upper bound
        U(mu) = max_{y >= 0} [<g, z - y> - gamma/2 ||y - z||^2 - mu^T s(y)]
                + sum_i tau w_i (exp(mu_i/tau) - 1),
    with w = s(z); U(mu) is tight at the optimal mu, which bisection finds.
    Any mu yields a valid bound, so an inexact bisection only loosens it.  For a
    single column the maximization is separable and has a closed form."""
    gamma = float(gamma)
    tau = float(tau)
    if gamma <= 0.0 or tau < 0.0:
        raise ValueError("The certificate needs gamma > 0 and tau >= 0.")
    g = np.asarray(gradient, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        g, z = g[:, None], z[:, None]
    rest = np.asarray(rest, dtype=np.float64).reshape(-1)
    w = np.maximum(rest + z.sum(axis=1), EPS)

    if tau > 0.0 and z.shape[1] == 1:
        # One entry per row: the maximization is separable and solved exactly.
        # Stationarity tau log(u) + gamma u = tau log(w) + gamma w - g for
        # u = rest + y, i.e. u = (tau/gamma) omega(t) with the Wright omega.
        g1, z1 = g[:, 0], z[:, 0]
        t = np.log(w) + (gamma * w - g1) / tau - np.log(tau / gamma)
        y = np.maximum((tau / gamma) * np.real(wrightomega(t)) - rest, 0.0)
        u = np.maximum(rest + y, 1e-300)
        step = z1 - y
        bound = (
            float(np.dot(g1, step))
            - 0.5 * gamma * float(np.dot(step, step))
            - tau * float(np.sum(u * np.log(u / w) - u + w))
        )
        return float(max(bound, 0.0))

    def row_sums(mu):
        y = np.maximum(z - (g + mu[:, None]) / gamma, 0.0)
        return y, rest + y.sum(axis=1)

    if tau > 0.0:
        def residual(mu):
            _, total = row_sums(mu)
            return tau * np.log(np.maximum(total, 1e-300) / w) - mu

        lo = -np.ones_like(w)
        hi = np.ones_like(w)
        for _ in range(200):
            low_bad = residual(lo) < 0.0
            if not low_bad.any():
                break
            lo[low_bad] *= 2.0
        for _ in range(200):
            high_bad = residual(hi) > 0.0
            if not high_bad.any():
                break
            hi[high_bad] *= 2.0
        for _ in range(int(bisection_steps)):
            mid = 0.5 * (lo + hi)
            positive = residual(mid) > 0.0
            lo = np.where(positive, mid, lo)
            hi = np.where(positive, hi, mid)
        mu = 0.5 * (lo + hi)
    else:
        mu = np.zeros_like(w)

    y, total = row_sums(mu)
    step = z - y
    bound = float(np.sum(g * step)) - 0.5 * gamma * float(np.sum(step * step))
    if tau > 0.0:
        bound += tau * float(np.sum(w * np.expm1(mu / tau))) - float(np.dot(mu, total))
    return float(max(bound, 0.0))


def entropy_block_gap(d, p, z, L, gamma):
    """Gap contribution of one column for the Bregman step p -> z (the same
    expression as the GS score in evaluate_block)."""
    return max(
        float(np.dot(d, p - z))
        + 0.5 * float(gamma) * float(np.dot(p, p) - np.dot(z, z))
        - float(L) * entropy_kl(z, p),
        0.0,
    )


def solve_selected_coordinate_subproblem_certified(
    problem,
    plan,
    source_marginal,
    target_marginal,
    j,
    selected_v,
    config,
    precomputed_candidate=None,
    max_inner=None,
):
    """Repeat the column's Bregman step until the subproblem is certified to
    accuracy eps_k, i.e. phi_hat(z) - min phi_hat <= eps_k as in
    eqn:epsilon_update, with

        eps_k = inner_rho * max(Delta_k, phi_hat(z_0) - phi_hat(z)),

    Delta_k being the selected column's gap contribution (the rule of
    eq:app-inner-stop in the PIP experiment).  The left-hand side is bounded
    by subproblem_certificate, so no reference minimizer is needed.  A
    solve also ends at max_inner steps or when the certificate falls below the
    floating-point resolution of phi_hat.

    Returns (column, source_marginal, steps, certificate_checks, converged)."""
    if max_inner is None:
        max_inner = config.max_inner_iterations
    max_inner = int(max_inner)
    gamma = problem.quadratic_weight
    shift = problem.sparsity_weight - selected_v

    p0 = np.maximum(plan[:, j], config.min_plan_value)
    obj0 = column_subproblem_objective(
        problem, p0, source_marginal, float(target_marginal[j]), j, selected_v
    )
    d0 = column_gradient_f(problem, source_marginal, float(target_marginal[j]), j) + shift
    L0 = columnwise_entropy_smoothness(problem, source_marginal, p0, config)
    if precomputed_candidate is None:
        z = entropy_bcdc_candidate(
            p0, d0, L0, gamma, config.min_plan_value, log_radius=config.block_log_radius
        )
    else:
        z = np.asarray(precomputed_candidate, dtype=np.float64)
    block_gap = entropy_block_gap(d0, p0, z, L0, gamma)
    src = source_marginal + (z - p0)
    steps = 1
    checks = 0
    converged = False
    floor = 8.0 * np.finfo(float).eps * (1.0 + abs(obj0))

    while True:
        tgt_j = float(z.sum())
        d = column_gradient_f(problem, src, tgt_j, j) + shift
        certificate = subproblem_certificate(
            d + gamma * z, z, src - z, problem.source_kl_weight, gamma
        )
        checks += 1
        decrease = obj0 - column_subproblem_objective(
            problem, z, src, tgt_j, j, selected_v
        )
        eps_k = config.inner_rho * max(block_gap, decrease)
        if certificate <= max(eps_k, floor):
            converged = True
            break
        if steps >= max_inner:
            break
        # The gradient used by the certificate is reused for the next step.
        L = columnwise_entropy_smoothness(problem, src, z, config)
        candidate = entropy_bcdc_candidate(
            z, d, L, gamma, config.min_plan_value, log_radius=config.block_log_radius
        )
        src = src + (candidate - z)
        z = candidate
        steps += 1

    return z, src, steps, checks, converged


def choose_block(
    problem,
    plan,
    source_marginal,
    target_marginal,
    config,
    rng,
    forced_uniform_block=None,
    forced_candidates=None,
):
    n = problem.num_target
    m = problem.num_source
    if config.selection_rule == "uniform":
        j = int(
            rng.integers(n) if forced_uniform_block is None else forced_uniform_block
        )
        score, candidate, _, _, selected_v = evaluate_block(
            problem,
            plan,
            source_marginal,
            target_marginal,
            j,
            config,
            need_candidate=True,
        )
        return j, selected_v, candidate, score, 0, 0

    batch_size = min(config.candidate_batch_size, n)
    if forced_candidates is None:
        candidates = rng.choice(n, size=batch_size, replace=False)
    else:
        candidates = np.asarray(forced_candidates, dtype=np.int64)
        if candidates.size == 0:
            raise ValueError("A GS candidate batch must contain at least one column.")
    need_all_candidates = config.selection_rule == "bregman_gap"

    # GS scoring touches candidate columns once for gradient information and
    # once for curvature/score information.
    selection_columns = int(candidates.size)
    selection_touched = int(2 * m * candidates.size)

    scores, candidates_z, _, _, selected_v = evaluate_block_batch(
        problem,
        plan,
        source_marginal,
        target_marginal,
        candidates,
        config,
        need_candidate=need_all_candidates,
    )
    best_pos = int(np.argmax(scores))
    best_j = int(candidates[best_pos])
    best_score = float(scores[best_pos])
    best_candidate = None if candidates_z is None else candidates_z[:, best_pos].copy()
    best_v = selected_v[:, best_pos].copy()

    return (
        best_j,
        best_v,
        best_candidate,
        best_score,
        selection_columns,
        selection_touched,
    )


def selected_topq_subgradient_matrix(problem, plan):
    v = np.zeros_like(plan, dtype=np.float64)
    for j in range(problem.num_target):
        v[topq_indices(plan[:, j], problem.top_k), j] = problem.sparsity_weight
    return v


def full_gradient_f(problem, source_marginal, target_marginal):
    r = np.maximum(source_marginal, EPS)
    s = np.maximum(target_marginal, EPS)
    return (
        problem.cost
        + problem.source_kl_weight * np.log(r / problem.source_mass)[:, None]
        + problem.target_kl_weight * np.log(s / problem.target_mass)[None, :]
    )


def full_selected_dc_terms_with_anchor(
    problem, plan, source_marginal, target_marginal, v_anchor
):
    d = (
        full_gradient_f(problem, source_marginal, target_marginal)
        + problem.sparsity_weight
        - v_anchor
    )
    selected_dc_gradient = d + problem.quadratic_weight * plan
    return d, selected_dc_gradient


def full_subproblem_objective(
    problem, plan, source_marginal, target_marginal, v_anchor
):
    return float(
        np.sum(problem.cost * plan)
        + problem.source_kl_weight
        * generalized_kl(source_marginal, problem.source_mass)
        + problem.target_kl_weight
        * generalized_kl(target_marginal, problem.target_mass)
        + 0.5 * problem.quadratic_weight * np.sum(plan * plan)
        + problem.sparsity_weight * np.sum(plan)
        - float(np.sum(v_anchor * plan))
    )


def full_entropy_candidate_with_anchor(
    problem, plan, source_marginal, target_marginal, v_anchor, config
):
    d, _ = full_selected_dc_terms_with_anchor(
        problem, plan, source_marginal, target_marginal, v_anchor
    )
    L = problem.entropy_relative_smoothness * config.relative_smoothness_scale
    return entropy_bcdc_candidate(
        plan,
        d,
        L,
        problem.quadratic_weight,
        config.min_plan_value,
    )


def full_euclidean_lipschitz_bound(problem, source_marginal, target_marginal):
    r_min = max(float(np.min(source_marginal)), EPS)
    s_min = max(float(np.min(target_marginal)), EPS)
    return float(
        problem.quadratic_weight
        + problem.source_kl_weight * problem.num_target / r_min
        + problem.target_kl_weight * problem.num_source / s_min
    )


def full_euclidean_candidate_with_anchor(
    problem, plan, source_marginal, target_marginal, v_anchor, config
):
    _, selected_dc_gradient = full_selected_dc_terms_with_anchor(
        problem, plan, source_marginal, target_marginal, v_anchor
    )
    ell = (
        full_euclidean_lipschitz_bound(problem, source_marginal, target_marginal)
        * config.relative_smoothness_scale
    )
    candidate = np.maximum(
        plan - selected_dc_gradient / max(ell, EPS),
        config.min_plan_value,
    )
    return candidate, float(ell)


BCDC_COMPARISON_RULES = ("lipschitz", "uniform", "bregman_gap")
FULL_COMPARISON_METHODS = ("full_entropy", "full_euclidean")
COMPARISON_METHODS = BCDC_COMPARISON_RULES + FULL_COMPARISON_METHODS

METHOD_LABELS = {
    "uniform": "Randomized entropy-BCDC",
    "gradient": "GS-gradient entropy-BCDC",
    "lipschitz": "GS-Lipschitz entropy-BCDC",
    "bregman_gap": "GS-non-Euclidean-gap entropy-BCDC",
    "full_entropy": "Full non-Euclidean DCA",
    "full_euclidean": "Full Euclidean DCA",
}


def resolve_method_integer_budgets(overrides, default, name, allowed_methods=None):
    allowed = tuple(COMPARISON_METHODS if allowed_methods is None else allowed_methods)
    overrides = {} if overrides is None else dict(overrides)
    unknown = sorted(set(overrides) - set(allowed))
    if unknown:
        raise ValueError(
            f"{name} contains unknown method keys: {unknown}. "
            f"Allowed keys are {list(allowed)}."
        )
    budgets = {}
    for method_key in allowed:
        value = overrides.get(method_key, default)
        if isinstance(value, bool):
            raise ValueError(f"{name}[{method_key!r}] must be a positive integer.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name}[{method_key!r}] must be positive.")
        budgets[method_key] = value
    return budgets


def build_method_config(
    base_config,
    method_key,
    outer_iterations_by_method,
    max_inner_iterations_by_method,
    inner_tol_by_method=None,
):
    if method_key not in COMPARISON_METHODS:
        raise ValueError(f"Unknown comparison method {method_key!r}.")
    cfg = replace(
        base_config,
        num_sweeps=int(outer_iterations_by_method[method_key]),
        max_inner_iterations=int(max_inner_iterations_by_method[method_key]),
    )
    if inner_tol_by_method is not None:
        # Per-method inner tolerance override. Without this, every method
        # shared a single global inner_tol, so tuning it for the coordinate
        # methods' adaptive stopping (e.g. via --coord-inner-tol) silently
        # loosened the full-dimensional methods' own inner-loop stopping
        # check too, cutting their inner solves short and making them look
        # far cheaper (and less converged) than intended.
        cfg = replace(cfg, inner_tol=float(inner_tol_by_method[method_key]))
    if method_key in BCDC_COMPARISON_RULES:
        cfg = replace(cfg, selection_rule=method_key)
    return cfg


def resolve_method_float_budgets(overrides, default, name, allowed_methods=None):
    allowed = tuple(COMPARISON_METHODS if allowed_methods is None else allowed_methods)
    overrides = {} if overrides is None else dict(overrides)
    unknown = sorted(set(overrides) - set(allowed))
    if unknown:
        raise ValueError(
            f"{name} contains unknown method keys: {unknown}. "
            f"Allowed keys are {list(allowed)}."
        )
    budgets = {}
    for method_key in allowed:
        value = float(overrides.get(method_key, default))
        if value < 0.0:
            raise ValueError(f"{name}[{method_key!r}] must be nonnegative.")
        budgets[method_key] = value
    return budgets


def make_method_budget_table(
    outer_iterations_by_method, max_inner_iterations_by_method, num_target
):
    rows = []
    for method_key in COMPARISON_METHODS:
        if method_key in BCDC_COMPARISON_RULES:
            budget_kind = "BCDC sweeps"
            coordinate_updates_per_outer = "num_target"
            total_coordinate_updates = int(
                outer_iterations_by_method[method_key]
            ) * int(num_target)
        else:
            budget_kind = "full DCA outer iterations"
            coordinate_updates_per_outer = "full matrix"
            total_coordinate_updates = np.nan
        rows.append(
            {
                "method": METHOD_LABELS[method_key],
                "method_key": method_key,
                "budget_kind": budget_kind,
                "configured_outer_iterations": int(
                    outer_iterations_by_method[method_key]
                ),
                "coordinate_updates_per_outer": coordinate_updates_per_outer,
                "configured_coordinate_updates": total_coordinate_updates,
                "configured_max_inner_iterations": int(
                    max_inner_iterations_by_method[method_key]
                ),
            }
        )
    return pd.DataFrame(rows)


def dense_problem_nnz(problem):
    return int(problem.num_source * problem.num_target)


def init_counters():
    return {
        "wall_clock_time": 0.0,
        "optimization_time": 0.0,
        "touched_nonzeros": 0,
        "column_accesses": 0,
        "objective_evaluations": 0,
        "mean_step_size": 0.0,
        "acceptance_rate": 1.0,
        "number_backtracking_steps": 0,
        "inner_solves": 0,
        "inner_solves_unconverged": 0,
        "certificate_checks": 0,
    }


def selected_subgradient_full_gap(
    problem, plan, source_marginal, target_marginal, config
):
    gap = 0.0
    diagnostic_config = replace(config, selection_rule="bregman_gap")
    for j in range(problem.num_target):
        _, _, _, block_gap, _ = evaluate_block(
            problem,
            plan,
            source_marginal,
            target_marginal,
            j,
            diagnostic_config,
            need_candidate=True,
        )
        gap += block_gap
    return float(max(gap, 0.0))


def selected_subgradient_kkt_residual(
    problem, plan, source_marginal, target_marginal, config
):
    squared = 0.0
    for j in range(problem.num_target):
        _, _, gradient, _, _ = evaluate_block(
            problem,
            plan,
            source_marginal,
            target_marginal,
            j,
            replace(config, selection_rule="gradient"),
            need_candidate=False,
        )
        projected = projected_block_gradient(
            gradient, plan[:, j], config.min_plan_value
        )
        squared += float(np.dot(projected, projected))
    return float(np.sqrt(squared))


def gap_concentration(problem, plan, source_marginal, target_marginal, config, log_radius):
    """Column gap contributions Delta_j at the current plan (each with the
    column's own top-Q subgradient) and Gamma = n max_j Delta_j / sum_j Delta_j
    of thm:existing-gap-gs (Gamma = 1 when the sum vanishes).  log_radius is the
    trust radius used for the constant L_j; np.inf gives the global constant
    tau_a + tau_b, i.e. the fixed Psi of the theory."""
    diagnostic_config = replace(
        config, selection_rule="bregman_gap", block_log_radius=log_radius
    )
    _, _, _, gaps, _ = evaluate_block_batch(
        problem,
        plan,
        source_marginal,
        target_marginal,
        np.arange(problem.num_target),
        diagnostic_config,
        need_candidate=True,
    )
    total = float(np.sum(gaps))
    largest = float(np.max(gaps))
    n = problem.num_target
    gamma_k = n * largest / total if total > 0.0 else 1.0
    # The experiments select the best of a random batch of b columns, so the
    # factor that matters for them is n E[max_{j in batch} Delta_j] / sum_j
    # Delta_j.  The r-th largest Delta is the batch maximum with probability
    # C(n - r, b - 1) / C(n, b), computed here in log space.
    b = min(int(config.candidate_batch_size), n)
    ranks = np.arange(1, n + 1)
    log_prob = np.full(n, -np.inf)
    valid = n - ranks >= b - 1
    log_prob[valid] = (
        gammaln(n - ranks[valid] + 1)
        - gammaln(b)
        - gammaln(n - ranks[valid] - b + 2)
        - (gammaln(n + 1) - gammaln(b + 1) - gammaln(n - b + 1))
    )
    expected_batch_max = float(np.dot(np.exp(log_prob), np.sort(gaps)[::-1]))
    gamma_batch = n * expected_batch_max / total if total > 0.0 else 1.0
    return total, largest, float(gamma_k), float(gamma_batch)


def make_history_row(
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
    source_residual = float(
        np.sum(np.abs(source_marginal - problem.source_mass))
        / np.sum(problem.source_mass)
    )
    target_residual = float(
        np.sum(np.abs(target_marginal - problem.target_mass))
        / np.sum(problem.target_mass)
    )
    dense_nnz = dense_problem_nnz(problem)
    matvec_pass_equivalent = (
        float(counters["touched_nonzeros"] / dense_nnz) if dense_nnz > 0 else np.nan
    )
    row = {
        "method": METHOD_LABELS[method_key],
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
        "objective": objective_value(problem, plan, source_marginal, target_marginal),
        "source_marginal_relative_l1_residual": source_residual,
        "target_marginal_relative_l1_residual": target_residual,
        "marginal_relative_l1_residual": max(source_residual, target_residual),
        "mean_step_size": float(counters["mean_step_size"]),
        "acceptance_rate": float(counters["acceptance_rate"]),
        "number_backtracking_steps": int(counters["number_backtracking_steps"]),
        "inner_solves": int(counters["inner_solves"]),
        "inner_solves_unconverged": int(counters["inner_solves_unconverged"]),
        "certificate_checks": int(counters["certificate_checks"]),
    }
    if config.log_gamma:
        for name, radius in (("local", config.block_log_radius), ("global", np.inf)):
            total, largest, gamma_k, gamma_batch = gap_concentration(
                problem, plan, source_marginal, target_marginal, config, radius
            )
            row[f"gap_sum_{name}"] = total
            row[f"gap_max_{name}"] = largest
            row[f"Gamma_{name}"] = gamma_k
            row[f"Gamma_batch_{name}"] = gamma_batch
    return row


def append_history_row(
    rows,
    problem,
    plan,
    source_marginal,
    target_marginal,
    config,
    iteration,
    sweep,
    counters,
    selected_score,
    t0,
    method_key=None,
    outer_iteration=None,
    inner_iterations=None,
    subproblem_obj_change=np.nan,
    subproblem_converged=False,
):
    counters["objective_evaluations"] += 1
    counters["wall_clock_time"] = time.perf_counter() - t0
    diagnostic_start = time.perf_counter()
    rows.append(
        make_history_row(
            problem,
            plan,
            source_marginal,
            target_marginal,
            config,
            iteration=iteration,
            sweep=sweep,
            counters=counters,
            selected_score=selected_score,
            method_key=method_key,
            outer_iteration=outer_iteration,
            inner_iterations=inner_iterations,
            subproblem_obj_change=subproblem_obj_change,
            subproblem_converged=subproblem_converged,
        )
    )
    return t0 + (time.perf_counter() - diagnostic_start)


def next_random_reshuffled_candidate_batch(rng, state, num_target, batch_size):
    num_target = int(num_target)
    batch_size = min(max(int(batch_size), 1), num_target)
    if state.get("order") is None or int(state.get("position", 0)) >= num_target:
        state["order"] = rng.permutation(num_target)
        state["position"] = 0

    start = int(state["position"])
    stop = min(start + batch_size, num_target)
    state["position"] = stop
    return np.asarray(state["order"][start:stop], dtype=np.int64)


def next_random_reshuffled_block(rng, state, num_target):
    return int(
        next_random_reshuffled_candidate_batch(rng, state, num_target, batch_size=1)[0]
    )


def solve_bcdc(
    problem,
    config,
    initial_plan=None,
    use_inner_iterations=False,
    inner_iterations_schedule=None,
    inner_usage_log=None,
):
    """use_inner_iterations=False (default) preserves the existing single-shot
    per-column update exactly as before. Set True to instead repeat the
    Bregman-proximal column update at each selected block, stopping early via
    config.inner_tol, matching how the full-dimensional methods and PIP
    already solve their frozen DCA subproblem -- i.e. this makes the
    coordinate method's inner accuracy match the outer accuracy condition of
    Algorithm 1 more closely, at additional per-block cost.

    By default (inner_iterations_schedule=None) the inner-iteration cap is
    flat: config.max_inner_iterations at every outer step. Pass a callable
    `inner_iterations_schedule(k) -> int`, where k is the 1-indexed outer
    (block-selection) iteration, to instead grow the inner budget with k --
    e.g. `lambda k: k` instantiates the t_k=k schedule discussed right after
    Theorem theorem:convergence in the paper (an O(1/t)-rate inner oracle run
    for t_k=k steps gives eps_k <= C/k, at a cost of sum_k k = O(K^2) total
    inner iterations -- only tractable for small K). Ignored when
    use_inner_iterations is False.

    If `inner_usage_log` is given a list, the number of inner iterations
    actually used at each outer (block-selection) step is appended to it
    in place -- a lightweight way to inspect how many inner steps an
    adaptive (config.inner_tol-driven) stopping rule ends up taking in
    practice, without changing the return signature.
    """
    config.validate(problem.num_target)
    plan = (
        initialize_plan(problem)
        if initial_plan is None
        else np.maximum(
            np.asarray(initial_plan, dtype=np.float64).copy(), config.min_plan_value
        )
    )
    source_marginal = plan.sum(axis=1)
    target_marginal = plan.sum(axis=0)
    rng = np.random.default_rng(config.seed)
    total_iterations = config.num_sweeps * problem.num_target
    record_every = config.record_every_sweeps * problem.num_target
    rows = []
    counters = init_counters()
    selected_score = np.nan
    total_inner_cumulative = 0
    uniform_state = {"order": None, "position": 0}
    candidate_batch_state = {"order": None, "position": 0}
    t0 = time.perf_counter()

    t0 = append_history_row(
        rows,
        problem,
        plan,
        source_marginal,
        target_marginal,
        config,
        iteration=0,
        sweep=0.0,
        counters=counters,
        selected_score=selected_score,
        t0=t0,
        outer_iteration=0,
        inner_iterations=0,
    )

    for iteration in range(1, total_iterations + 1):
        forced_uniform_block = None
        forced_candidates = None
        if config.sampling == "random_reshuffling":
            if config.selection_rule == "uniform":
                forced_uniform_block = next_random_reshuffled_block(
                    rng, uniform_state, problem.num_target
                )
            else:
                forced_candidates = next_random_reshuffled_candidate_batch(
                    rng,
                    candidate_batch_state,
                    problem.num_target,
                    config.candidate_batch_size,
                )

        update_start = time.perf_counter()
        (
            j,
            selected_v,
            precomputed_candidate,
            selected_score,
            selection_columns,
            selection_touched,
        ) = choose_block(
            problem,
            plan,
            source_marginal,
            target_marginal,
            config,
            rng,
            forced_uniform_block=forced_uniform_block,
            forced_candidates=forced_candidates,
        )
        counters["column_accesses"] += int(selection_columns)
        counters["touched_nonzeros"] += int(selection_touched)

        if config.inner_stopping == "certificate":
            step_max_inner = (
                int(inner_iterations_schedule(iteration))
                if inner_iterations_schedule is not None
                else None
            )
            candidate, updated_source_marginal, inner_used, checks, subproblem_converged = (
                solve_selected_coordinate_subproblem_certified(
                    problem,
                    plan,
                    source_marginal,
                    target_marginal,
                    j,
                    selected_v,
                    config,
                    precomputed_candidate=precomputed_candidate,
                    max_inner=step_max_inner,
                )
            )
            plan[:, j] = candidate
            source_marginal = updated_source_marginal
            target_marginal[j] = float(candidate.sum())
            total_inner_cumulative += int(inner_used)
            counters["inner_solves"] += 1
            counters["inner_solves_unconverged"] += int(not subproblem_converged)
            counters["certificate_checks"] += int(checks)
            # Every certificate needs the column gradient; all but the last one
            # are reused by the next step, but charge each check one column
            # pass anyway.
            counters["touched_nonzeros"] += int(problem.num_source * checks)
            if inner_usage_log is not None:
                inner_usage_log.append(int(inner_used))
        elif use_inner_iterations:
            step_max_inner = (
                int(inner_iterations_schedule(iteration))
                if inner_iterations_schedule is not None
                else None
            )
            candidate, updated_source_marginal, inner_used, subproblem_converged = (
                solve_selected_coordinate_subproblem_inner_loop(
                    problem,
                    plan,
                    source_marginal,
                    target_marginal,
                    j,
                    selected_v,
                    config,
                    precomputed_candidate=precomputed_candidate,
                    max_inner=step_max_inner,
                )
            )
            old = plan[:, j].copy()
            plan[:, j] = candidate
            source_marginal = updated_source_marginal
            target_marginal[j] = float(candidate.sum())
            total_inner_cumulative += int(inner_used)
            if inner_usage_log is not None:
                inner_usage_log.append(int(inner_used))
        else:
            candidate = solve_selected_coordinate_subproblem(
                problem,
                plan,
                source_marginal,
                target_marginal,
                j,
                selected_v,
                config,
                precomputed_candidate=precomputed_candidate,
            )
            old = plan[:, j].copy()
            delta = candidate - old
            plan[:, j] = candidate
            source_marginal += delta
            target_marginal[j] = float(candidate.sum())
            inner_used = 1
            total_inner_cumulative += 1
        counters["optimization_time"] += time.perf_counter() - update_start

        # Charge the gradient, candidate construction, and incremental marginal
        # update over the selected dense column -- once per inner step actually
        # performed, so repeated inner solves are charged fairly.
        counters["column_accesses"] += int(inner_used)
        counters["touched_nonzeros"] += int(3 * problem.num_source * inner_used)

        if iteration % record_every == 0 or iteration == total_iterations:
            sweep = iteration / problem.num_target
            t0 = append_history_row(
                rows,
                problem,
                plan,
                source_marginal,
                target_marginal,
                config,
                iteration=iteration,
                sweep=sweep,
                counters=counters,
                selected_score=selected_score,
                t0=t0,
                outer_iteration=int(np.ceil(sweep)),
                inner_iterations=total_inner_cumulative,
            )

    return plan, pd.DataFrame(rows)


def solve_full_dca(problem, config, initial_plan=None, geometry="entropy"):
    if geometry not in {"entropy", "euclidean"}:
        raise ValueError("geometry must be 'entropy' or 'euclidean'.")
    config.validate(problem.num_target)
    plan = (
        initialize_plan(problem)
        if initial_plan is None
        else np.maximum(
            np.asarray(initial_plan, dtype=np.float64).copy(), config.min_plan_value
        )
    )
    source_marginal = plan.sum(axis=1)
    target_marginal = plan.sum(axis=0)
    method_key = "full_entropy" if geometry == "entropy" else "full_euclidean"
    rows = []
    counters = init_counters()
    selected_score = np.nan
    total_inner = 0
    subproblem_obj_change = np.nan
    subproblem_converged = False
    dense_nnz = dense_problem_nnz(problem)
    t0 = time.perf_counter()

    t0 = append_history_row(
        rows,
        problem,
        plan,
        source_marginal,
        target_marginal,
        config,
        iteration=0,
        sweep=0.0,
        counters=counters,
        selected_score=selected_score,
        t0=t0,
        method_key=method_key,
        outer_iteration=0,
        inner_iterations=0,
        subproblem_obj_change=subproblem_obj_change,
        subproblem_converged=subproblem_converged,
    )

    for outer in range(1, config.num_sweeps + 1):
        v_anchor = selected_topq_subgradient_matrix(problem, plan)
        previous_subproblem = full_subproblem_objective(
            problem, plan, source_marginal, target_marginal, v_anchor
        )
        subproblem_converged = False
        inner = 0
        mean_step_sum = 0.0
        certified = config.inner_stopping == "certificate"
        objective_start = previous_subproblem
        floor = 8.0 * np.finfo(float).eps * (1.0 + abs(objective_start))
        full_gap = 0.0

        for inner in range(1, int(config.max_inner_iterations) + 1):
            update_start = time.perf_counter()
            if geometry == "entropy":
                candidate = full_entropy_candidate_with_anchor(
                    problem, plan, source_marginal, target_marginal, v_anchor, config
                )
                selected_score = np.nan
                mean_step = 1.0 / (
                    problem.entropy_relative_smoothness
                    * config.relative_smoothness_scale
                )
            else:
                candidate, selected_score = full_euclidean_candidate_with_anchor(
                    problem, plan, source_marginal, target_marginal, v_anchor, config
                )
                mean_step = 1.0 / max(float(selected_score), EPS)

            if certified and inner == 1:
                # Gap of the full block at the outer iterate, from the first
                # step (Delta_k in eps_k; recomputed here, charged with the step).
                d0, g0 = full_selected_dc_terms_with_anchor(
                    problem, plan, source_marginal, target_marginal, v_anchor
                )
                if geometry == "entropy":
                    L = problem.entropy_relative_smoothness * config.relative_smoothness_scale
                    full_gap = (
                        float(np.sum(d0 * (plan - candidate)))
                        + 0.5
                        * problem.quadratic_weight
                        * float(np.sum(plan * plan) - np.sum(candidate * candidate))
                        - L * entropy_kl(candidate, plan)
                    )
                else:
                    step = plan - candidate
                    full_gap = float(np.sum(g0 * step)) - 0.5 * float(
                        selected_score
                    ) * float(np.sum(step * step))
                full_gap = max(full_gap, 0.0)

            plan = candidate
            source_marginal = plan.sum(axis=1)
            target_marginal = plan.sum(axis=0)
            counters["optimization_time"] += time.perf_counter() - update_start

            # Charge the full gradient, curvature/model information, and full
            # matrix/marginal refresh.
            counters["touched_nonzeros"] += int(3 * dense_nnz)
            counters["column_accesses"] += int(problem.num_target)
            mean_step_sum += float(mean_step)
            counters["mean_step_size"] = mean_step_sum / inner

            current_subproblem = full_subproblem_objective(
                problem, plan, source_marginal, target_marginal, v_anchor
            )
            subproblem_obj_change = relative_objective_change(
                previous_subproblem, current_subproblem
            )
            previous_subproblem = current_subproblem
            if certified:
                _, gradient = full_selected_dc_terms_with_anchor(
                    problem, plan, source_marginal, target_marginal, v_anchor
                )
                certificate = subproblem_certificate(
                    gradient,
                    plan,
                    np.zeros(problem.num_source),
                    problem.source_kl_weight,
                    problem.quadratic_weight,
                )
                counters["certificate_checks"] += 1
                counters["touched_nonzeros"] += int(dense_nnz)
                eps_k = config.inner_rho * max(
                    full_gap, objective_start - current_subproblem
                )
                if certificate <= max(eps_k, floor):
                    subproblem_converged = True
                    break
            elif subproblem_obj_change <= config.inner_tol:
                subproblem_converged = True
                break

        total_inner += int(inner)
        counters["inner_solves"] += 1
        counters["inner_solves_unconverged"] += int(not subproblem_converged)

        if outer % config.record_every_sweeps == 0 or outer == config.num_sweeps:
            t0 = append_history_row(
                rows,
                problem,
                plan,
                source_marginal,
                target_marginal,
                config,
                iteration=total_inner,
                sweep=float(total_inner),
                counters=counters,
                selected_score=selected_score,
                t0=t0,
                method_key=method_key,
                outer_iteration=outer,
                inner_iterations=total_inner,
                subproblem_obj_change=subproblem_obj_change,
                subproblem_converged=subproblem_converged,
            )

    return plan, pd.DataFrame(rows)
