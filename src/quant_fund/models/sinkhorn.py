"""Sinkhorn canon (Cuturi 2013): entropic optimal transport
via log-domain matrix scaling, returning the optimal
transport plan and regularized cost, checked against the
exact transport LP optimum as epsilon shrinks.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

FloatArray = np.ndarray


def sinkhorn(
    a: FloatArray,
    b: FloatArray,
    cost: FloatArray,
    eps: float = 0.05,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> tuple[FloatArray, float]:
    """Log-domain Sinkhorn; returns (plan, transport cost <C,P>)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    k = -cost / eps

    def _logsumexp(x: FloatArray, axis: int) -> FloatArray:
        mx = x.max(axis=axis, keepdims=True)
        return np.asarray(
            mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True)),
            dtype=np.float64,
        )

    for _ in range(max_iter):
        u_prev = u.copy()
        u = log_a - _logsumexp(k + v[None, :], axis=1).ravel()
        v = log_b - _logsumexp(k + u[:, None], axis=0).ravel()
        if np.max(np.abs(u - u_prev)) < tol:
            break
    log_p = k + u[:, None] + v[None, :]
    p = np.exp(log_p)
    return p, float(np.sum(cost * p))


def exact_transport(a: FloatArray, b: FloatArray, cost: FloatArray) -> tuple[FloatArray, float]:
    """Exact transport LP via HiGHS (small instances only)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    m, n = cost.shape
    a_eq = np.zeros((m + n, m * n))
    for i in range(m):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n):
        a_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([a, b])
    res = linprog(
        cost.ravel(),
        A_eq=a_eq[:-1],
        b_eq=b_eq[:-1],
        bounds=(0.0, None),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"transport LP failed: {res.message}")
    return res.x.reshape(m, n), float(res.fun)


def bench_sinkhorn(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 12
    xs = rng.random((n, 1))
    xt = rng.random((n, 1))
    cost = (xs - xt.T) ** 2
    a = rng.dirichlet(np.ones(n))
    b = rng.dirichlet(np.ones(n))
    p, c = sinkhorn(a, b, cost, eps=0.01)
    marg = float(
        max(
            np.max(np.abs(p.sum(axis=1) - a)),
            np.max(np.abs(p.sum(axis=0) - b)),
        )
    )
    _, c_coarse = sinkhorn(a, b, cost, eps=0.1)
    _, c_fine = sinkhorn(a, b, cost, eps=0.002)
    _, lp = exact_transport(a, b, cost)
    return {
        "synthetic_marginal_err": marg,
        "synthetic_cost_eps_coarse": c_coarse,
        "synthetic_cost_eps_mid": c,
        "synthetic_cost_eps_fine": c_fine,
        "synthetic_lp_cost": lp,
        "synthetic_fine_gap": c_fine - lp,
    }
