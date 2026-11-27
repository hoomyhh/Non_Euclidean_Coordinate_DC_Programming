"""Block-coordinate DCA for sparse unbalanced optimal transport.

The problem is

    min_{P >= 0}  <C, P> + tau_a KL(P1 | a) + tau_b KL(P^T 1 | b)
                  + gamma/2 ||P||_F^2 + lam * sum_j (||p_j||_1 - ||p_j||_(Q)),

with the DC split f = <C, P> + tau_a KL(P1 | a) + tau_b KL(P^T 1 | b),
g = gamma/2 ||P||_F^2 + lam ||P||_1 and h = lam * sum_j ||p_j||_(Q), where
||p||_(Q) is the sum of the Q largest entries.  The blocks are the columns
of P.  Each subproblem (one column, or the full plan) is solved by repeated
Bregman steps until its accuracy is certified (subproblem_certificate).

Work is counted in matvec-pass equivalents: one pass touches all m*n
entries of the plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import wrightomega

EPS = 1e-12

METHODS = ("kl_random", "kl_gs_breg", "kl_gs_pg", "kl_full", "e_full")
LABELS = {
    "kl_random": "KL-Random",
    "kl_gs_breg": "KL-GS(Breg)",
    "kl_gs_pg": "KL-GS(PG)",
    "kl_full": "KL-Full",
    "e_full": "E-Full",
}


@dataclass(frozen=True)
class Problem:
    cost: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tau_a: float
    tau_b: float
    gamma: float
    lam: float
    q: int

    @property
    def m(self):
        return int(self.cost.shape[0])

    @property
    def n(self):
        return int(self.cost.shape[1])


@dataclass(frozen=True)
class Settings:
    sweeps: int  # column methods: sweeps of n updates; full methods: outer iterations
    max_steps: int = 200  # Bregman steps per subproblem
    theta: float = 0.1  # eps_k = theta * max(Delta_k, decrease so far)
    batch_size: int = 8  # Gauss-Southwell candidate batch
    log_radius: float = 4.0  # column trust region |log(z_i / p_ij)| <= log_radius
    min_value: float = 1e-300
    order_seed: int = 0  # random-reshuffled column order


# ---------------------------------------------------------------- objective


def generalized_kl(x, y):
    x = np.maximum(np.asarray(x, dtype=np.float64), EPS)
    y = np.maximum(np.asarray(y, dtype=np.float64), EPS)
    return float(np.sum(x * np.log(x / y) - x + y))


def entropy_kl(z, p):
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
    return float(np.sum(z * np.log(z / p) - z + p))


def entropy_kl_columns(z, p):
    z = np.maximum(np.asarray(z, dtype=np.float64), 1e-300)
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-300)
    return np.sum(z * np.log(z / p) - z + p, axis=0)


def topq_indices(x, q):
    q = min(max(int(q), 0), x.size)
    if q == 0:
        return np.empty(0, dtype=np.int64)
    if q == x.size:
        return np.arange(x.size, dtype=np.int64)
    return np.argpartition(x, -q)[-q:]


def objective(problem, plan, r, s):
    trimmed = sum(
        float(plan[:, j].sum()) - float(np.sum(plan[topq_indices(plan[:, j], problem.q), j]))
        for j in range(problem.n)
    )
    return float(
        np.sum(problem.cost * plan)
        + problem.tau_a * generalized_kl(r, problem.a)
        + problem.tau_b * generalized_kl(s, problem.b)
        + 0.5 * problem.gamma * np.sum(plan * plan)
        + problem.lam * trimmed
    )


def initial_plan(problem):
    plan = np.outer(problem.a, problem.b) / float(problem.a.sum())
    return np.maximum(plan, EPS)


# ------------------------------------------------------------ column blocks


def topq_column_subgradient(p, q, lam):
    """lam * indicator of the Q largest entries of the column p."""
    v = np.zeros_like(p, dtype=np.float64)
    v[topq_indices(p, q)] = lam
    return v


def topq_subgradient(values, q, lam):
    """lam * indicator of the Q largest entries of each column of `values`."""
    values = np.asarray(values, dtype=np.float64)
    m, k = values.shape
    q = min(max(int(q), 0), m)
    v = np.zeros_like(values)
    if q == 0:
        return v
    if q == m:
        v[:, :] = lam
        return v
    rows = np.argpartition(values, -q, axis=0)[-q:, :]
    v[rows, np.arange(k)[None, :]] = lam
    return v


def column_gradient(problem, r, s_j, j):
    """Gradient of f with respect to column j."""
    r = np.maximum(r, EPS)
    s_j = max(float(s_j), EPS)
    return (
        problem.cost[:, j]
        + problem.tau_a * np.log(r / problem.a)
        + problem.tau_b * np.log(s_j / problem.b[j])
    )


def columns_gradient(problem, r, s, cols):
    r = np.maximum(r, EPS)
    s_cols = np.maximum(s[cols], EPS)
    return (
        problem.cost[:, cols]
        + problem.tau_a * np.log(r / problem.a)[:, None]
        + problem.tau_b * np.log(s_cols / problem.b[cols])[None, :]
    )


def column_subproblem(problem, z, r, s_j, j, v_j):
    """phi_hat for column j up to a constant, with r the source marginal that
    already contains z."""
    r = np.maximum(r, EPS)
    s_j = max(float(s_j), EPS)
    return float(
        np.dot(problem.cost[:, j], z)
        + problem.tau_a * generalized_kl(r, problem.a)
        + problem.tau_b * generalized_kl(np.array([s_j]), np.array([problem.b[j]]))
        + 0.5 * problem.gamma * float(np.dot(z, z))
        + problem.lam * float(np.sum(z))
        - float(np.dot(v_j, z))
    )


def trust_region(p, log_radius, min_value):
    if log_radius is None or not np.isfinite(float(log_radius)):
        return None, None
    p = np.maximum(np.asarray(p, dtype=np.float64), min_value)
    lower = np.maximum(p * np.exp(-float(log_radius)), min_value)
    upper = np.maximum(p * np.exp(float(log_radius)), lower)
    return lower, upper


def entropy_step(p, d, L, gamma, min_value, log_radius=None):
    """Minimizer of <d, z> + gamma/2 ||z||^2 + L KL(z | p), entrywise
    gamma z + L log(z / p) + d = 0, i.e. a Lambert-W (Wright omega) formula,
    clipped to the trust region."""
    p = np.maximum(np.asarray(p, dtype=np.float64), min_value)
    d = np.asarray(d, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    gamma = float(gamma)
    if gamma == 0.0:
        log_z = np.log(p) - d / L
        z = np.exp(np.clip(log_z, np.log(min_value), np.log(np.finfo(float).max)))
    else:
        z = (L / gamma) * wrightomega(np.log(gamma * p / L) - d / L)
    lower, upper = trust_region(p, log_radius, min_value)
    if lower is not None:
        z = np.minimum(np.maximum(z, lower), upper)
    return np.maximum(np.asarray(z, dtype=np.float64), min_value)


def column_constant(problem, r, p, settings):
    """Columnwise relative-smoothness constant L_j on the trust region
    (appendix of the paper)."""
    alpha = np.exp(float(settings.log_radius))
    r = np.maximum(np.asarray(r, dtype=np.float64), EPS)
    p = np.maximum(np.asarray(p, dtype=np.float64), settings.min_value)
    fraction = np.clip(alpha * p / np.maximum(r + np.expm1(float(settings.log_radius)) * p, EPS), 0.0, 1.0)
    return float(problem.tau_b + problem.tau_a * float(np.max(fraction)))


def column_constants(problem, r, p, settings):
    alpha = np.exp(float(settings.log_radius))
    r = np.maximum(np.asarray(r, dtype=np.float64), EPS)[:, None]
    p = np.maximum(np.asarray(p, dtype=np.float64), settings.min_value)
    fraction = np.clip(alpha * p / np.maximum(r + np.expm1(float(settings.log_radius)) * p, EPS), 0.0, 1.0)
    return np.asarray(problem.tau_b + problem.tau_a * np.max(fraction, axis=0), dtype=np.float64)


def column_gap(d, p, z, L, gamma):
    """Gap contribution of a column for the Bregman step p -> z."""
    return max(
        float(np.dot(d, p - z))
        + 0.5 * float(gamma) * float(np.dot(p, p) - np.dot(z, z))
        - float(L) * entropy_kl(z, p),
        0.0,
    )


# -------------------------------------------------------------- certificate


def subproblem_certificate(gradient, z, rest, tau, gamma, bisection_steps=60):
    """Upper bound on phi_hat(z) - min_{y >= 0} phi_hat(y) (appendix, `Stopping
    rule').

    Write phi_hat = H + R + linear, with H(y) = tau KL(rest + row sums of y | a)
    + gamma/2 ||y||^2 and R the target KL, which is convex.  Then
    phi_hat(z) - min phi_hat <= max_{y >= 0} <g, z - y> - D_H(y, z).

    z is one column (rest = row sums of the other columns), where the
    maximization is separable and has a closed form, or the full plan
    (rest = 0), where the row sums couple the entries.  In the latter case the
    row KL is written through its conjugate; every multiplier mu then gives a
    separable upper bound, minimized over mu by bisection."""
    gamma = float(gamma)
    tau = float(tau)
    g = np.asarray(gradient, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        g, z = g[:, None], z[:, None]
    rest = np.asarray(rest, dtype=np.float64).reshape(-1)
    w = np.maximum(rest + z.sum(axis=1), EPS)

    if z.shape[1] == 1:
        # tau log(u) + gamma u = tau log(w) + gamma w - g for u = rest + y.
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

    def residual(mu):
        _, total = row_sums(mu)
        return tau * np.log(np.maximum(total, 1e-300) / w) - mu

    lo = -np.ones_like(w)
    hi = np.ones_like(w)
    for _ in range(200):
        bad = residual(lo) < 0.0
        if not bad.any():
            break
        lo[bad] *= 2.0
    for _ in range(200):
        bad = residual(hi) > 0.0
        if not bad.any():
            break
        hi[bad] *= 2.0
    for _ in range(int(bisection_steps)):
        mid = 0.5 * (lo + hi)
        positive = residual(mid) > 0.0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)
    mu = 0.5 * (lo + hi)

    y, total = row_sums(mu)
    step = z - y
    bound = float(np.sum(g * step)) - 0.5 * gamma * float(np.sum(step * step))
    bound += tau * float(np.sum(w * np.expm1(mu / tau))) - float(np.dot(mu, total))
    return float(max(bound, 0.0))


def solve_column(problem, plan, r, s, j, v_j, settings, first_step=None):
    """Solve the subproblem of column j to the certified accuracy

        phi_hat(z) - min phi_hat <= theta * max(Delta, phi_hat(z0) - phi_hat(z)),

    where Delta is the column's gap contribution.  Returns the column, the
    updated source marginal, the number of steps and certificate evaluations,
    and whether the accuracy was certified within max_steps."""
    gamma = problem.gamma
    shift = problem.lam - v_j
    p0 = np.maximum(plan[:, j], settings.min_value)
    obj0 = column_subproblem(problem, p0, r, float(s[j]), j, v_j)
    d0 = column_gradient(problem, r, float(s[j]), j) + shift
    L0 = column_constant(problem, r, p0, settings)
    if first_step is None:
        z = entropy_step(p0, d0, L0, gamma, settings.min_value, settings.log_radius)
    else:
        z = np.asarray(first_step, dtype=np.float64)
    delta = column_gap(d0, p0, z, L0, gamma)
    r = r + (z - p0)
    steps, checks, certified = 1, 0, False
    floor = 8.0 * np.finfo(float).eps * (1.0 + abs(obj0))

    while True:
        s_j = float(z.sum())
        d = column_gradient(problem, r, s_j, j) + shift
        bound = subproblem_certificate(d + gamma * z, z, r - z, problem.tau_a, gamma)
        checks += 1
        decrease = obj0 - column_subproblem(problem, z, r, s_j, j, v_j)
        if bound <= max(settings.theta * max(delta, decrease), floor):
            certified = True
            break
        if steps >= settings.max_steps:
            break
        L = column_constant(problem, r, z, settings)
        z_next = entropy_step(z, d, L, gamma, settings.min_value, settings.log_radius)
        r = r + (z_next - z)
        z = z_next
        steps += 1

    return z, r, steps, checks, certified


def select_column(problem, plan, r, s, method, cols, settings):
    """Score the candidate columns `cols` and return the selected column, its
    subgradient block, the first Bregman step if it was computed, and the
    number of entries touched by the scoring."""
    m = problem.m
    if method == "kl_random":
        j = int(cols[0])
        p = plan[:, j]
        v_j = topq_column_subgradient(p, problem.q, problem.lam)
        d = column_gradient(problem, r, float(s[j]), j) + problem.lam - v_j
        L = column_constant(problem, r, p, settings)
        z = entropy_step(p, d, L, problem.gamma, settings.min_value, settings.log_radius)
        return j, v_j, z, 0

    p = plan[:, cols]
    v = topq_subgradient(p, problem.q, problem.lam)
    d = columns_gradient(problem, r, s, cols) + problem.lam - v
    first_steps = None
    if method == "kl_gs_breg":
        # Predicted Bregman decrease: the gap contribution of each column.
        L = column_constants(problem, r, p, settings)
        first_steps = entropy_step(p, d, L, problem.gamma, settings.min_value, settings.log_radius)
        scores = (
            np.sum(d * (p - first_steps), axis=0)
            + 0.5 * problem.gamma * (np.sum(p * p, axis=0) - np.sum(first_steps * first_steps, axis=0))
            - L * entropy_kl_columns(first_steps, p)
        )
        scores = np.maximum(scores.astype(np.float64), 0.0)
    else:
        # Projected gradient normalized by a local Euclidean Lipschitz bound.
        grad = d + problem.gamma * p
        grad[(p <= settings.min_value * (1.0 + 1e-8)) & (grad > 0.0)] = 0.0
        ell = (
            problem.gamma
            + problem.tau_a / max(float(np.min(r)), EPS)
            + problem.tau_b * problem.m / np.maximum(s[cols], EPS)
        )
        scores = np.linalg.norm(grad, axis=0) / np.sqrt(np.maximum(ell, EPS))
    best = int(np.argmax(scores))
    first_step = None if first_steps is None else first_steps[:, best].copy()
    return int(cols[best]), v[:, best].copy(), first_step, 2 * m * len(cols)


# ------------------------------------------------------------------ solvers


class History:
    """Records objective, marginal infeasibility and work along a run."""

    def __init__(self, problem, method):
        self.problem, self.method = problem, method
        self.rows = []
        self.touched = 0
        self.solves = self.uncertified = self.steps = 0
        self.time = 0.0

    def record(self, plan, r, s, outer):
        problem = self.problem
        source = float(np.sum(np.abs(r - problem.a)) / np.sum(problem.a))
        target = float(np.sum(np.abs(s - problem.b)) / np.sum(problem.b))
        self.rows.append(
            {
                "method": LABELS[self.method],
                "outer_iteration": int(outer),
                "matvec": self.touched / (problem.m * problem.n),
                "objective": objective(problem, plan, r, s),
                "infeasibility": max(source, target),
                "subproblems": self.solves,
                "uncertified": self.uncertified,
                "bregman_steps": self.steps,
                "time_seconds": self.time,
            }
        )

    def frame(self):
        return pd.DataFrame(self.rows)


def solve_columns(problem, settings, method):
    """Block-coordinate DCA over the columns with random-reshuffled selection:
    KL-Random updates the columns in a random order, the Gauss-Southwell
    variants pick the best of consecutive batches of that order."""
    plan = np.maximum(initial_plan(problem), settings.min_value)
    r, s = plan.sum(axis=1), plan.sum(axis=0)
    rng = np.random.default_rng(settings.order_seed)
    history = History(problem, method)
    history.record(plan, r, s, 0)
    batch = 1 if method == "kl_random" else min(settings.batch_size, problem.n)
    order, position = None, problem.n

    for it in range(1, settings.sweeps * problem.n + 1):
        if position >= problem.n:
            order, position = rng.permutation(problem.n), 0
        cols = np.asarray(order[position : position + batch], dtype=np.int64)
        position += len(cols)

        start = time.perf_counter()
        j, v_j, first_step, scoring = select_column(problem, plan, r, s, method, cols, settings)
        z, r, steps, checks, certified = solve_column(
            problem, plan, r, s, j, v_j, settings, first_step
        )
        plan[:, j] = z
        s[j] = float(z.sum())
        history.time += time.perf_counter() - start

        # Each Bregman step costs 3m (gradient, step, marginal update), each
        # certificate evaluation m; Gauss-Southwell scoring costs 2m per candidate.
        history.touched += scoring + 3 * problem.m * steps + problem.m * checks
        history.solves += 1
        history.uncertified += int(not certified)
        history.steps += steps
        if it % problem.n == 0:
            history.record(plan, r, s, it // problem.n)

    return plan, history.frame()


def full_terms(problem, plan, r, s, v):
    """d = grad f - v + lam and the gradient of phi_hat, d + gamma P."""
    d = (
        problem.cost
        + problem.tau_a * np.log(np.maximum(r, EPS) / problem.a)[:, None]
        + problem.tau_b * np.log(np.maximum(s, EPS) / problem.b)[None, :]
        + problem.lam
        - v
    )
    return d, d + problem.gamma * plan


def full_subproblem(problem, plan, r, s, v):
    return float(
        np.sum(problem.cost * plan)
        + problem.tau_a * generalized_kl(r, problem.a)
        + problem.tau_b * generalized_kl(s, problem.b)
        + 0.5 * problem.gamma * np.sum(plan * plan)
        + problem.lam * np.sum(plan)
        - float(np.sum(v * plan))
    )


def solve_full(problem, settings, method):
    """Full DCA: freeze the top-Q subgradient of every column and solve the
    full subproblem by entropy steps with the global constant tau_a + tau_b
    (KL-Full) or by projected-gradient steps with a Euclidean Lipschitz bound
    (E-Full), until the accuracy is certified as for the columns."""
    plan = np.maximum(initial_plan(problem), settings.min_value)
    r, s = plan.sum(axis=1), plan.sum(axis=0)
    mn = problem.m * problem.n
    history = History(problem, method)
    history.record(plan, r, s, 0)
    L = float(problem.tau_a + problem.tau_b)

    for outer in range(1, settings.sweeps + 1):
        v = np.zeros_like(plan, dtype=np.float64)
        for j in range(problem.n):
            v[:, j] = topq_column_subgradient(plan[:, j], problem.q, problem.lam)
        obj0 = full_subproblem(problem, plan, r, s, v)
        floor = 8.0 * np.finfo(float).eps * (1.0 + abs(obj0))
        delta, certified, steps = 0.0, False, 0

        for steps in range(1, settings.max_steps + 1):
            start = time.perf_counter()
            d, grad = full_terms(problem, plan, r, s, v)
            if method == "kl_full":
                z = entropy_step(plan, d, L, problem.gamma, settings.min_value)
            else:
                ell = float(
                    problem.gamma
                    + problem.tau_a * problem.n / max(float(np.min(r)), EPS)
                    + problem.tau_b * problem.m / max(float(np.min(s)), EPS)
                )
                z = np.maximum(plan - grad / max(ell, EPS), settings.min_value)
            if steps == 1:
                if method == "kl_full":
                    delta = (
                        float(np.sum(d * (plan - z)))
                        + 0.5 * problem.gamma * float(np.sum(plan * plan) - np.sum(z * z))
                        - L * entropy_kl(z, plan)
                    )
                else:
                    step = plan - z
                    delta = float(np.sum(grad * step)) - 0.5 * ell * float(np.sum(step * step))
                delta = max(delta, 0.0)
            plan = z
            r, s = plan.sum(axis=1), plan.sum(axis=0)
            history.time += time.perf_counter() - start

            _, grad = full_terms(problem, plan, r, s, v)
            bound = subproblem_certificate(grad, plan, np.zeros(problem.m), problem.tau_a, problem.gamma)
            history.touched += 3 * mn + mn
            decrease = obj0 - full_subproblem(problem, plan, r, s, v)
            if bound <= max(settings.theta * max(delta, decrease), floor):
                certified = True
                break

        history.solves += 1
        history.uncertified += int(not certified)
        history.steps += steps
        history.record(plan, r, s, outer)

    return plan, history.frame()


def solve(problem, settings, method):
    if method in ("kl_full", "e_full"):
        return solve_full(problem, settings, method)
    if method in ("kl_random", "kl_gs_breg", "kl_gs_pg"):
        return solve_columns(problem, settings, method)
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
