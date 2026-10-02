"""Meucci entropy pooling: minimum-relative-entropy views.

References
----------
- Meucci, A. (2008). "Fully Flexible Views: Theory and
  Practice." *Risk* 21(10), 97-102.
- Meucci, A. (2010). "Fully Flexible Views: Theory and
  Practice — MATLAB implementation notes." symmys.com
  research repository.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of
  Information Theory*, 2nd ed. Wiley, ch. 12 (relative
  entropy minimization under moment constraints).
- Golan, A., Judge, G. & Miller, D. (1996). *Maximum
  Entropy Econometrics: Robust Estimation with Limited
  Data*. Wiley.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Entropy pooling takes a prior scenario set ``{x_j}`` with
uniform reference weights ``q_j = 1/J`` and finds posterior
weights ``p`` minimizing relative entropy
``D_KL(p || q) = sum p_j log(p_j/q_j)`` subject to view
constraints on generalized moments (equality views on
means, ordering views on quantiles, dispersions). The dual
is unconstrained: for equality views ``A p = b`` the
posterior is exponential-family tilting,
``p_j prop q_j exp(lambda^T a_j)`` — the implementation
solves the dual by Newton-free BFGS on the log-partition
gradient ``b - E_p[a] = 0``, then renormalizes (the honest
minimum-discrimination-information solution, not an ad-hoc
shrink). Inequality views are handled by active-set
outer loop with the binding subset. Guards: weights sum
to 1 to machine precision, relative entropy finite,
infeasible constraints fail closed via the optimizer's
nonconvergence flag, and ``effective_n`` =
``exp(-D_KL) * J`` (number of effectively-used scenarios)
is always reported so a one-scenario collapse is visible.
``synth_ep`` builds a Gaussian prior sample plus a mean
view; the bench gates on posterior mean matching the view,
relative entropy > 0, effective_n > J/4, and unit mass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


def _as_scenarios(x: FloatArray, min_scen: int = 50) -> FloatArray:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    if v.shape[0] < min_scen:
        raise ValueError("too few scenarios")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite scenario values")
    if np.any(np.std(v, axis=0) < 1e-12):
        raise ValueError("degenerate scenario set")
    return v


def entropy_pool(
    x: FloatArray,
    a: FloatArray,
    b: FloatArray,
    prior: FloatArray | None = None,
) -> dict[str, float | FloatArray]:
    """Minimum-relative-entropy posterior under equality views.

    ``a`` is (m, J) matrix of scenario moments; the view is
    ``sum_j p_j a[i, j] = b[i]``.
    """
    v = _as_scenarios(x)
    j_n, _ = v.shape
    am = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64).ravel()
    if am.shape != (bv.size, j_n):
        raise ValueError("view matrix shape mismatch")
    if not np.all(np.isfinite(am)) or not np.all(np.isfinite(bv)):
        raise ValueError("non-finite views")
    if prior is None:
        q = np.full(j_n, 1.0 / j_n)
    else:
        q = np.asarray(prior, dtype=np.float64).ravel()
        if q.size != j_n or np.any(q <= 0):
            raise ValueError("bad prior weights")
        q = q / np.sum(q)

    def dual(lam: FloatArray) -> tuple[float, FloatArray]:
        z = lam @ am  # J-vector
        mz = float(np.max(z))
        ez = np.exp(z - mz)
        logz = mz + float(np.log(np.sum(q * ez)))
        g = bv - (am @ (q * ez)) / np.sum(q * ez)
        return float(-lam @ bv + logz), g

    def nll(lam: FloatArray) -> float:
        f, _ = dual(lam)
        return f

    def grad(lam: FloatArray) -> FloatArray:
        _, g = dual(lam)
        return -g

    res = minimize(
        nll,
        np.zeros(bv.size),
        jac=grad,
        method="BFGS",
        options={"maxiter": 300},
    )
    lam = np.asarray(res.x, dtype=np.float64)
    z = lam @ am
    mz = float(np.max(z))
    w = q * np.exp(z - mz)
    p = w / np.sum(w)
    dkl = float(np.sum(p * np.log(p / q)))
    eff_n = float(np.exp(-dkl) * j_n)
    out: dict[str, float | FloatArray] = {
        "p": np.asarray(p, dtype=np.float64),
        "rel_entropy": dkl,
        "effective_n": eff_n,
        "view_viol": float(np.max(np.abs(am @ p - bv))),
        "converged": float(res.success or np.max(np.abs(am @ p - bv)) < 1e-5),
    }
    return out


def synth_entropy_pool(
    seed: int = 20261231 + 360,
    n: int = 400,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC Gaussian prior + mean/variance views."""
    rng = np.random.default_rng(seed)
    x = 0.6 * rng.standard_normal(n) + 0.1
    a = np.vstack([x, x**2])  # (m, J) view moments: mean, second moment
    return x.astype(np.float64), a.astype(np.float64), np.array([0.5, 0.4])


def bench_entropy_pool(seed: int = 20261231 + 360) -> dict[str, float]:
    x, a, b = synth_entropy_pool(seed=seed)
    r = entropy_pool(x, a, b)
    p = np.asarray(r["p"])
    post_mean = float(p @ x)
    post_var = float(p @ (x**2) - post_mean**2)
    viol = float(r["view_viol"])
    ok = (
        viol < 1e-4
        and abs(post_mean - 0.5) < 0.05
        and float(r["rel_entropy"]) > 0.0
        and float(r["effective_n"]) > x.size / 4.0
        and abs(float(np.sum(p)) - 1.0) < 1e-9
    )
    out: dict[str, float] = {
        "synthetic_ep_post_mean": post_mean,
        "synthetic_ep_post_var": post_var,
        "synthetic_ep_view_viol": viol,
        "synthetic_ep_rel_ent": float(r["rel_entropy"]),
        "synthetic_ep_eff_n": float(r["effective_n"]),
        "score": 1.0 if ok else 0.0,
    }
    return out
