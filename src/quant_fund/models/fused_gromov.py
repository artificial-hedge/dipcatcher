"""Fused Gromov-Wasserstein canon (Vayer, Chapel, Flamary, (SYNTHETIC)
Tavenard & Fournier 2019): FGW mixes a feature-space
Wasserstein term with the structural GW term,
alpha*<M,P> + (1-alpha)*GW — on synthetic attributed
metric-measure spaces.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.gromov_wasserstein import _sinkhorn_plan, _tensor_cost

FloatArray = np.ndarray


def fused_gromov(
    f1: FloatArray,
    f2: FloatArray,
    c1: FloatArray,
    c2: FloatArray,
    p: FloatArray,
    q: FloatArray,
    alpha: float = 0.5,
    eps: float = 0.02,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> tuple[FloatArray, float]:
    """Entropic FGW; returns (plan, FGW objective)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m_feat = np.asarray(
        (np.sum(f1**2, axis=1, keepdims=True) + np.sum(f2**2, axis=1)[None, :]) - 2.0 * f1 @ f2.T,
        dtype=np.float64,
    )
    t = np.outer(p, q)
    prev = np.inf
    for _ in range(max_iter):
        gw_cost = _tensor_cost(c1, c2, t)
        cost = alpha * m_feat + (1.0 - alpha) * gw_cost
        t = _sinkhorn_plan(p, q, cost, eps)
        obj = float(np.sum(cost * t))
        if abs(prev - obj) < tol * max(1.0, abs(prev)):
            break
        prev = obj
    gw_cost = _tensor_cost(c1, c2, t)
    fgw = float(np.sum((alpha * m_feat + (1.0 - alpha) * gw_cost) * t))
    return t, fgw


def bench_fused_gromov(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 10
    x = rng.random((n, 3))
    c1 = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))
    perm = rng.permutation(n)
    c2 = c1[np.ix_(perm, perm)]
    f1 = x[:, :1]
    f2 = f1[perm]  # features aligned under same permutation
    p = np.full(n, 1.0 / n)
    q = np.full(n, 1.0 / n)
    t_f, fgw_f = fused_gromov(f1, f2, c1, c2, p, q, alpha=0.5, eps=0.005, max_iter=40)
    # shuffled features: FGW should rise since features no longer align
    t_s, fgw_s = fused_gromov(
        f1, f1[rng.permutation(n)], c1, c2, p, q, alpha=0.5, eps=0.005, max_iter=40
    )
    marg = float(max(np.max(np.abs(t_f.sum(1) - p)), np.max(np.abs(t_f.sum(0) - q))))
    return {
        "synthetic_fgw_aligned": fgw_f,
        "synthetic_fgw_shuffled": fgw_s,
        "synthetic_marginal_err": marg,
        "synthetic_sep": fgw_s - fgw_f,
    }
