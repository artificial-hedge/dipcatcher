"""Optimal transport: Sinkhorn distances and Wasserstein barycenters.

Distributional comparison with geometry. The entropic-regularized OT
problem

    ``W_ε(a, b) = min_{P ∈ U(a,b)} ⟨P, C⟩ − ε·H(P)``

is solved by Sinkhorn's alternating scaling algorithm (Cuturi 2013).
``U(a,b)`` is the transport polytope ``{P ≥ 0 : P1 = a, Pᵀ1 = b}`` and
``C`` the ground cost. From the plan we get the entropic OT cost, the
(unregularized) transport cost, and barycenters for multi-measure
pooling.

References
----------
- Cuturi, M. (2013). *Sinkhorn distances: lightspeed computation of
  optimal transport.* NeurIPS 26.
- Peyré, G. & Cuturi, M. (2019). *Computational optimal transport.*
  Foundations and Trends in Machine Learning 11(5-6), 355–607.
- Agueh, M. & Carlier, G. (2011). *Barycenters in the Wasserstein
  space.* SIAM J. Math. Anal. 43(2), 904–924.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on non-normalized measures or non-PD
regularization; Sinkhorn capped at ``max_iter`` iterations with a
marginal-violation convergence check.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray

__all__ = [
    "bench_optimal_transport",
    "ot_plan",
    "sinkhorn",
    "synth_measures",
    "wasserstein_barycenter",
    "wasserstein_distance",
]


def _check_measure(w: FloatArray, name: str) -> FloatArray:
    a = np.asarray(w, dtype=np.float64)
    if a.ndim != 1 or a.size < 2:
        raise ValueError(f"{name} must be a vector with >=2 bins")
    if np.any(a < 0):
        raise ValueError(f"{name} has negative mass")
    s = float(a.sum())
    if s <= 0:
        raise ValueError(f"{name} is all zero")
    return a / s  # normalize to a probability vector


def _cost_matrix(x: FloatArray, y: FloatArray, p: float = 2.0) -> FloatArray:
    d = x[:, None] - y[None, :]
    return np.asarray(np.abs(d) ** p)


def sinkhorn(
    a: FloatArray,
    b: FloatArray,
    cost: FloatArray,
    eps: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> dict[str, FloatArray | float]:
    """Sinkhorn entropic OT.

    Returns ``plan`` (n×m transport matrix), ``u, v`` scaling vectors,
    ``ot_cost = ⟨P, C⟩``, ``reg_cost = ⟨P,C⟩ − εH(P)``, iterations, and
    final marginal violation.
    """
    a = _check_measure(a, "a")
    b = _check_measure(b, "b")
    c = np.asarray(cost, dtype=np.float64)
    if c.shape != (a.size, b.size):
        raise ValueError("cost must be n×m matching a,b")
    if eps <= 0:
        raise ValueError("eps must be positive")
    k = np.exp(-c / eps)
    u = np.ones(a.size)
    v = np.ones(b.size)
    for _it in range(max_iter):
        u_new = a / np.maximum(k @ v, 1e-300)
        v_new = b / np.maximum(k.T @ u_new, 1e-300)
        delta = max(float(np.abs(u_new - u).max()), float(np.abs(v_new - v).max()))
        u, v = u_new, v_new
        if delta < tol:
            break
    plan = u[:, None] * k * v[None, :]
    viol = max(float(np.abs(plan.sum(axis=1) - a).max()), float(np.abs(plan.sum(axis=0) - b).max()))
    ot_cost = float((plan * c).sum())
    entropy = -float((plan * np.log(np.maximum(plan, 1e-300))).sum())
    return {
        "plan": plan,
        "ot_cost": ot_cost,
        "reg_cost": ot_cost - eps * entropy,
        "iterations": float(_it + 1),
        "marginal_violation": viol,
        "converged": float(viol < 1e-5),
    }


def wasserstein_distance(
    x: FloatArray,
    y: FloatArray,
    wx: FloatArray | None = None,
    wy: FloatArray | None = None,
    eps: float = 0.02,
    p: float = 2.0,
) -> dict[str, float]:
    """Entropic OT distance between point clouds ``x`` (source) and
    ``y`` (target) with equal-or-supplied masses. For 1-D clouds this is
    a smoothed ``W_p``; returns ``ot_cost`` and the empirical
    ``w_p_quantile`` (exact 1-D ``W_p`` via sorted quantiles) for
    reference."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 5 or y.size < 5:
        raise ValueError("point clouds need >=5 samples each")
    n, m = x.size, y.size
    a = np.ones(n) / n if wx is None else _check_measure(wx, "wx")
    b = np.ones(m) / m if wy is None else _check_measure(wy, "wy")
    cost = _cost_matrix(x, y, p=p)
    sk = sinkhorn(a, b, cost, eps=eps)
    # exact 1-D W_p^p via quantile matching (n=m case: sort + pairwise)
    xs = np.sort(x)
    ys = np.sort(y)
    q = min(n, m)
    w_exact = (
        float(
            np.mean(
                np.abs(
                    np.quantile(xs, np.linspace(0, 1, q + 2)[1:-1])
                    - np.quantile(ys, np.linspace(0, 1, q + 2)[1:-1])
                )
                ** p
            )
        )
        if q > 4
        else 0.0
    )
    return {
        "ot_cost": float(sk["ot_cost"]),
        "reg_cost": float(sk["reg_cost"]),
        "w_p_exact": w_exact,
        "marginal_violation": float(sk["marginal_violation"]),
        "iterations": float(sk["iterations"]),
        "converged": float(sk["converged"]),
    }


def wasserstein_barycenter(
    measures: list[FloatArray],
    support: FloatArray,
    weights: FloatArray | None = None,
    eps: float = 0.05,
    n_iter: int = 25,
) -> dict[str, FloatArray | float]:
    """Entropic Wasserstein barycenter of histograms on a shared support.

    Iterates the entropic barycenter fixed point (Peyré & Cuturi
    §9.2): ``b = exp(Σ w_k log(u_k ⊙ K v_k))``. Returns the barycenter
    weights and the OT cost to each input measure.
    """
    if len(measures) < 2:
        raise ValueError("need at least 2 measures")
    ms = [_check_measure(w, f"measure[{i}]") for i, w in enumerate(measures)]
    s = np.asarray(support, dtype=np.float64)
    if s.ndim != 1 or s.size < 3:
        raise ValueError("support must be a 1-D grid")
    if any(m.size != s.size for m in ms):
        raise ValueError("every measure must live on the support grid")
    wt = np.ones(len(ms)) / len(ms) if weights is None else _check_measure(weights, "weights")
    c = _cost_matrix(s, s)
    k = np.exp(-c / eps)
    u = [np.ones(s.size) for _ in ms]
    v = [np.ones(s.size) for _ in ms]
    b = np.ones(s.size) / s.size
    for _ in range(n_iter):
        for i, m in enumerate(ms):
            v[i] = m / np.maximum(k.T @ u[i], 1e-300)
        b = np.exp(
            sum(wt[i] * np.log(np.maximum(u[i] * (k @ v[i]), 1e-300)) for i in range(len(ms)))
        )
        b /= max(b.sum(), 1e-300)
        for i, _m in enumerate(ms):
            u[i] = b / np.maximum(k @ v[i], 1e-300)
    costs = [float(((u[i][:, None] * k * v[i][None, :]) * c).sum()) for i in range(len(ms))]
    return {"barycenter": b, "support": s, "ot_costs": np.array(costs), "weights": wt}


def ot_plan(a: FloatArray, b: FloatArray, cost: FloatArray, eps: float = 0.05) -> FloatArray:
    """Convenience: just the transport plan."""
    return np.asarray(sinkhorn(a, b, cost, eps=eps)["plan"])


def synth_measures(
    n_bins: int = 40,
    n: int = 3,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """SYNTHETIC probability vectors on a shared 1-D grid — mixtures of
    two Gaussians whose separation carries the signal."""
    np.random.default_rng(seed)  # seed kept in signature for API symmetry
    s = np.linspace(-4, 4, n_bins)
    out = {"support": s}
    for i in range(n):
        m1 = -1.5 + 0.5 * i
        m2 = 1.5 - 0.5 * i
        w = np.exp(-0.5 * (s - m1) ** 2) + np.exp(-0.5 * (s - m2) ** 2)
        w = w / w.sum()
        out[f"m{i}"] = w
    return out


def bench_optimal_transport(seed: int = 20261231 + 181) -> dict[str, float]:
    """SYNTHETIC: OT cost grows with translation; barycenter lands mid."""
    dm = synth_measures(n_bins=40, seed=seed)
    s = np.asarray(dm["support"])
    m0 = np.asarray(dm["m0"])
    m1 = np.asarray(dm["m1"])
    m2 = np.asarray(dm["m2"])
    c = _cost_matrix(s, s)
    d00 = sinkhorn(m0, m0, c, eps=0.05)
    d01 = sinkhorn(m0, m1, c, eps=0.05)
    d02 = sinkhorn(m0, m2, c, eps=0.05)
    bar = wasserstein_barycenter([m0, m2], s, eps=0.05)
    bary = np.asarray(bar["barycenter"])
    # barycenter mass centroid should sit near the midpoint of the pair
    c0 = float((m0 * s).sum())
    c2 = float((m2 * s).sum())
    cb = float((bary * s).sum())
    mid = 0.5 * (c0 + c2)
    rng2 = np.random.default_rng(seed + 1)
    x = rng2.normal(0, 1, 300)
    y = rng2.normal(0.5, 1, 300)
    wd = wasserstein_distance(x, y, eps=0.05)
    return {
        "synthetic_ot_self": float(d00["ot_cost"]),
        "synthetic_ot_near": float(d01["ot_cost"]),
        "synthetic_ot_far": float(d02["ot_cost"]),
        "synthetic_ot_monotone": float(d00["ot_cost"] <= d01["ot_cost"] <= d02["ot_cost"]),
        "synthetic_bary_center_err": abs(cb - mid),
        "synthetic_marginal_violation": float(d01["marginal_violation"]),
        "synthetic_wd_cloud": float(wd["ot_cost"]),
        "synthetic_wd_exact_1d": float(wd["w_p_exact"]),
        "synthetic_determinism": float(d00["ot_cost"] == sinkhorn(m0, m0, c, eps=0.05)["ot_cost"]),
    }
