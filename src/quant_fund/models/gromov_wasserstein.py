"""Gromov-Wasserstein canon (Mémoli 2011, Peyré-Cuturi-
Solomon 2016): entropic GW distance between metric-measure
spaces via nested Sinkhorn iterations on the quadratic
cost tensor, verified on isomorphic and perturbed
synthetic mm-spaces.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def _sinkhorn_plan(a: FloatArray, b: FloatArray, cost: FloatArray, eps: float) -> FloatArray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    k = np.exp(-cost / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(300):
        u_prev = u
        u = a / np.maximum(k @ v, 1e-300)
        v = b / np.maximum(k.T @ u, 1e-300)
        if np.max(np.abs(u - u_prev)) < 1e-10:
            break
    return np.asarray((u[:, None] * k) * v[None, :], dtype=np.float64)


def _tensor_cost(c1: FloatArray, c2: FloatArray, t: FloatArray) -> FloatArray:
    """L(C1,C2) ⊗ T = sum over j,l of |C1[i,j]-C2[k,l]|^2 T[j,l]."""
    # c2_const[j,l] broadcast over (i,k)
    sq = (c1[:, None, :, None] - c2[None, :, None, :]) ** 2
    return np.asarray(np.einsum("ijkl,jl->ik", sq, t), dtype=np.float64)


def gromov_wasserstein(
    c1: FloatArray,
    c2: FloatArray,
    p: FloatArray,
    q: FloatArray,
    eps: float = 0.02,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> tuple[FloatArray, float]:
    """Entropic GW; returns (plan, GW discrepancy sqrt value)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    t = np.outer(p, q)
    prev = np.inf
    for _ in range(max_iter):
        cost = _tensor_cost(c1, c2, t)
        t = _sinkhorn_plan(p, q, cost, eps)
        obj = float(np.sum(cost * t))
        if abs(prev - obj) < tol * max(1.0, abs(prev)):
            break
        prev = obj
    cost = _tensor_cost(c1, c2, t)
    gw = float(np.sum(cost * t))
    return t, float(np.sqrt(max(gw, 0.0)))


def bench_gromov_wasserstein(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 10
    # space 1: random metric space; space 2: isomorphic (permuted)
    x = rng.random((n, 3))
    c1 = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))
    perm = rng.permutation(n)
    c2 = c1[np.ix_(perm, perm)]
    p = np.full(n, 1.0 / n)
    q = np.full(n, 1.0 / n)
    t_iso, gw_iso = gromov_wasserstein(c1, c2, p, q, eps=0.005, max_iter=40)
    # space 3: different geometry
    y = rng.random((n, 3)) * 3.0
    c3 = np.sqrt(((y[:, None] - y[None, :]) ** 2).sum(-1))
    _, gw_diff = gromov_wasserstein(c1, c3, p, q, eps=0.005, max_iter=40)
    return {
        "synthetic_gw_iso": gw_iso,
        "synthetic_gw_diff": gw_diff,
        "synthetic_iso_marginal": float(
            max(
                np.max(np.abs(t_iso.sum(1) - p)),
                np.max(np.abs(t_iso.sum(0) - q)),
            )
        ),
    }
