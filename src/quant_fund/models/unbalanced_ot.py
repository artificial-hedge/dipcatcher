"""Unbalanced optimal transport canon (Chizat, Peyré, (SYNTHETIC)
Schmitzer & Vialard 2018): KL-relaxed Sinkhorn where
marginals are penalized rather than constrained, so mass
can be created/destroyed — validated on synthetic measures
of unequal total mass.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def unbalanced_sinkhorn(
    a: FloatArray,
    b: FloatArray,
    cost: FloatArray,
    eps: float = 0.05,
    rho: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> tuple[FloatArray, float]:
    """KL-UOT Sinkhorn: u^(rho/(rho+eps)) scaling; returns (plan, cost)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    k = np.exp(-cost / eps)
    fi = rho / (rho + eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u.copy()
        u = (a / np.maximum(k @ v, 1e-300)) ** fi
        v = (b / np.maximum(k.T @ u, 1e-300)) ** fi
        if np.max(np.abs(u - u_prev)) < tol:
            break
    p = (u[:, None] * k) * v[None, :]
    return p, float(np.sum(cost * p))


def bench_unbalanced_ot(seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    n = 14
    xs = rng.random((n, 1))
    xt = rng.random((n, 1))
    cost = (xs - xt.T) ** 2
    a = rng.dirichlet(np.ones(n)) * 1.0
    b = rng.dirichlet(np.ones(n)) * 1.6  # 60% more mass on target
    p, c = unbalanced_sinkhorn(a, b, cost, eps=0.02, rho=0.5)
    mass_in = float(p.sum(axis=0).sum())
    marg_resid = float(np.mean(np.abs(p.sum(axis=1) - a)) + np.mean(np.abs(p.sum(axis=0) - b)))
    # tighter rho -> closer to balanced marginals (up to mass mismatch)
    p2, c2 = unbalanced_sinkhorn(a, b, cost, eps=0.02, rho=5.0)
    marg2 = float(np.mean(np.abs(p2.sum(axis=1) - a)) + np.mean(np.abs(p2.sum(axis=0) - b)))
    return {
        "synthetic_uot_cost": c,
        "synthetic_uot_cost_rho_hi": c2,
        "synthetic_mass_total": mass_in,
        "synthetic_marg_resid_rho05": marg_resid,
        "synthetic_marg_resid_rho5": marg2,
    }
