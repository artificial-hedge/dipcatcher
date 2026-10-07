"""Black-Litterman equilibrium + views posterior.

References
----------
- Black, F. & Litterman, R. (1992). "Global Portfolio
  Optimization." *Financial Analysts Journal* 48(5),
  28-43.
- He, G. & Litterman, R. (1999). "The Intuition Behind
  Black-Litterman Model Portfolios." Goldman Sachs
  Investment Management Research working paper.
- Idzorek, T. (2005). "A Step-by-Step Guide to the
  Black-Litterman Model." Zephyr Associates working
  paper.
- Satchell, S. & Scowcroft, A. (2000). "A Demystification
  of the Black-Litterman Model." *Journal of Asset
  Management* 1(2), 138-150.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The prior is reverse-optimized equilibrium: starting from
a reference portfolio ``w_eq``, implied returns are
``pi = delta * Sigma * w_eq``. Investor views enter as
``P mu ~ N(q, Omega)``; the posterior mean (He-Litterman
master formula) is
``mu_BL = [(tau Sigma)^{-1} + P' Omega^{-1} P]^{-1}
         [(tau Sigma)^{-1} pi + P' Omega^{-1} q]``
and the posterior covariance ``Sigma_BL =
[(tau Sigma)^{-1} + P' Omega^{-1} P]^{-1}`` — the honest
Bayesian combination, not a tilting heuristic. Omega is
set by the Idzorek-style proportional rule
``Omega_i = tau * p_i Sigma p_i'`` (view uncertainty tied
to the variance of the view itself) when not supplied.
The subtlety: with no views, mu_BL = pi exactly — the
model must collapse to equilibrium, which we gate on;
with dogmatic views (Omega -> 0) mu_BL -> the view.
``synth_bl`` builds a 3-asset equilibrium, applies one
absolute view (+200bp on asset 1), and gates on the
posterior tilting the right direction with bounded
deviation, exact collapse under no views, and PSD
posterior covariance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_inputs(
    sigma: FloatArray,
    w_eq: FloatArray,
    tau: float,
) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(w_eq, dtype=np.float64).ravel()
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] != w.size:
        raise ValueError("sigma/w_eq shape mismatch")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        raise ValueError("non-finite inputs")
    if tau <= 0.0 or float(np.min(np.linalg.eigvalsh(s))) < -1e-10:
        raise ValueError("bad tau or non-PSD sigma")
    return s, w


def implied_returns(
    sigma: FloatArray,
    w_eq: FloatArray,
    delta: float = 2.5,
) -> FloatArray:
    """Reverse-optimized equilibrium returns pi = delta Sigma w."""
    s, w = _check_inputs(sigma, w_eq, 1.0)
    return np.asarray(delta * s @ w, dtype=np.float64)


def bl_posterior(
    sigma: FloatArray,
    w_eq: FloatArray,
    p: FloatArray,
    q: FloatArray,
    tau: float = 0.05,
    delta: float = 2.5,
    omega: FloatArray | None = None,
) -> dict[str, float | FloatArray]:
    """Black-Litterman posterior under views P mu ~ N(q, Omega)."""
    s, w = _check_inputs(sigma, w_eq, tau)
    n = s.shape[0]
    pm = np.asarray(p, dtype=np.float64)
    if pm.ndim == 1:
        pm = pm[None, :]
    qv = np.asarray(q, dtype=np.float64).ravel()
    if pm.shape[0] != qv.size or pm.shape[1] != n:
        raise ValueError("view matrix shape mismatch")
    pi = implied_returns(s, w, delta)
    ts_inv = np.linalg.inv(tau * s)
    if omega is None:
        # Idzorek: view variance tied to view portfolio variance
        om = np.diag([tau * float(pm[i] @ s @ pm[i]) for i in range(qv.size)])
        om = np.asarray(om, dtype=np.float64)
    else:
        om = np.asarray(omega, dtype=np.float64)
    if om.shape != (qv.size, qv.size):
        raise ValueError("omega shape mismatch")
    om_inv = np.linalg.inv(om)
    post_cov = np.linalg.inv(ts_inv + pm.T @ om_inv @ pm)
    mu_bl = post_cov @ (ts_inv @ pi + pm.T @ om_inv @ qv)
    out: dict[str, float | FloatArray] = {
        "pi": np.asarray(pi, dtype=np.float64),
        "mu_bl": np.asarray(mu_bl, dtype=np.float64),
        "post_cov": np.asarray(post_cov, dtype=np.float64),
        "view_pull": np.asarray(mu_bl - pi, dtype=np.float64),
        "max_eig_neg": float(min(0.0, float(np.min(np.linalg.eigvalsh(post_cov))))),
    }
    return out


def synth_bl(seed: int = 20261231 + 365) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC 3-asset equilibrium + one absolute view."""
    chol = np.linalg.cholesky(
        np.array([[0.04, 0.01, 0.005], [0.01, 0.06, 0.008], [0.005, 0.008, 0.05]])
    )
    sigma = chol @ chol.T
    w_eq = np.array([0.5, 0.3, 0.2])
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([0.08])  # view: asset 0 earns 8%
    return (
        sigma.astype(np.float64),
        w_eq.astype(np.float64),
        p.astype(np.float64),
        q.astype(np.float64),
    )


def bench_bl(seed: int = 20261231 + 365) -> dict[str, float]:
    sigma, w_eq, p, q = synth_bl(seed=seed)
    r = bl_posterior(sigma, w_eq, p, q)
    pi = np.asarray(r["pi"])
    mu = np.asarray(r["mu_bl"])
    pull = np.asarray(r["view_pull"])
    # no-view collapse
    r_noview = bl_posterior(sigma, w_eq, np.zeros((0, 3)), np.zeros(0))
    collapse = float(np.max(np.abs(np.asarray(r_noview["mu_bl"]) - pi)))
    ok = (
        pull[0] > 0.005  # posterior tilts asset-0 return up toward view
        and mu[0] > pi[0]
        and mu[0] < q[0] + 0.01  # but doesn't fully hit the dogmatic view
        and collapse < 1e-10
        and float(r["max_eig_neg"]) <= 1e-12
    )
    out: dict[str, float] = {
        "synthetic_bl_pi0": float(pi[0]),
        "synthetic_bl_mu0": float(mu[0]),
        "synthetic_bl_pull0": float(pull[0]),
        "synthetic_bl_collapse_err": collapse,
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
