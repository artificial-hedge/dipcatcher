"""Euler risk allocation — Tasche marginal ES/VaR contributions.

For a portfolio loss L = Σ w_i X_i, the Euler allocation of a
homogeneous risk measure ρ splits the total into per-position
contributions that sum exactly to the whole:

    ρ(L) = Σ_i w_i · ∂ρ/∂w_i

For Value-at-Risk: ∂VaR_α/∂w_i = E[X_i | L = q_α] (smoothed with a
band around the quantile). For expected shortfall the clean
conditional form holds exactly:

    ES_α = E[L | L ≥ q_α] →  ES_i = w_i · E[X_i | L ≥ q_α]

so Σ_i ES_i = ES identically — the Euler property. Inputs are a
scenario matrix (n_scenarios, n_assets) of losses and a weight
vector; VaR uses a Gaussian kernel smoother on |L − q_α|.

References
----------
- Tasche, D. (1999/2008). "Risk contributions and performance
  measurement." / "Capital allocation to business units and
  sub-portfolios." *J. Economic Dynamics & Control* 32.
- McNeil, A.J., Frey, R., Embrechts, P. (2015). *Quantitative
  Risk Management*, 2nd ed. — §8.4 Euler allocation.
- Kalkbrener, M. (2005). "An axiomatic approach to capital
  allocation." *Mathematical Finance* 15.
- Meucci, A. (2007). "Risk contributions from generic
  user-defined factors." *Risk* 20(6).

Honesty
-------
SYNTHETIC Gaussian scenarios only; bench verifies the Euler
identity and recovers analytic Gaussian contributions — not a
capital-adequacy claim.

Composition
-----------
Called by ``quant_fund.research.benches_w67.bench_euler_risk``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_inputs(x: FloatArray, w: FloatArray) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.ndim != 2 or w.ndim != 1 or x.shape[1] != w.size:
        raise ValueError("x must be (n_scen, n_assets) matching w")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)):
        raise ValueError("non-finite inputs")
    return x, w


def expected_shortfall(x: FloatArray, w: FloatArray, alpha: float = 0.95) -> float:
    """ES_alpha of the portfolio loss w @ x over scenarios."""
    x, w = _check_inputs(x, w)
    if not (0.5 < alpha < 0.9999):
        raise ValueError("alpha out of range")
    loss = x @ w
    q = np.quantile(loss, alpha)
    tail = loss[loss >= q - 1e-14]
    return float(tail.mean())


def euler_es_contributions(x: FloatArray, w: FloatArray, alpha: float = 0.95) -> FloatArray:
    """Per-asset ES contributions ES_i = w_i E[X_i | L >= q_alpha];
    sums exactly to ES."""
    x, w = _check_inputs(x, w)
    if not (0.5 < alpha < 0.9999):
        raise ValueError("alpha out of range")
    loss = x @ w
    q = np.quantile(loss, alpha)
    mask = loss >= q - 1e-14
    return w * x[mask].mean(axis=0)


def euler_var_contributions(
    x: FloatArray, w: FloatArray, alpha: float = 0.95, h_frac: float = 0.05
) -> FloatArray:
    """Per-asset VaR contributions via kernel-smoothed conditional
    expectation around the quantile: w_i E[X_i | L ≈ q_alpha]."""
    x, w = _check_inputs(x, w)
    if not (0.5 < alpha < 0.9999):
        raise ValueError("alpha out of range")
    loss = x @ w
    q = float(np.quantile(loss, alpha))
    h = float(h_frac * np.std(loss))
    kern = np.exp(-0.5 * ((loss - q) / h) ** 2)
    kern = np.maximum(kern, 1e-12)
    cond_mean = (x * kern[:, None]).sum(axis=0) / kern.sum()
    return np.asarray(w * cond_mean, dtype=np.float64)


def bench_euler_risk(seed: int = 20261231 + 395) -> dict[str, float]:
    """SYNTHETIC check — Euler identity + Gaussian contributions."""
    rng = np.random.default_rng(seed)
    n_scen, n_assets = 40000, 4
    sd = np.array([0.8, 1.0, 1.5, 0.5])
    rho = 0.3
    corr = np.eye(n_assets) * (1 - rho) + rho
    cov = np.outer(sd, sd) * corr
    x = rng.multivariate_normal(np.zeros(n_assets), cov, n_scen)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    alpha = 0.95

    es = expected_shortfall(x, w, alpha)
    contrib = euler_es_contributions(x, w, alpha)
    euler_err = abs(contrib.sum() - es) / max(es, 1e-9)
    if euler_err > 1e-8:
        raise ValueError("Euler identity violated")
    # Analytic Gaussian ES contributions: ES = (w^T Σ) * phi(z)/ (1-α)
    # scaled by 1/sqrt(wΣw); contrib_i = w_i (Σw)_i / sqrt(wΣw) * φ/(1−α).
    from scipy.stats import norm

    sig_p = float(np.sqrt(w @ cov @ w))
    z = float(norm.ppf(alpha))
    analytic = w * (cov @ w) / sig_p * float(norm.pdf(z)) / (1 - alpha)
    rel_err = float(np.linalg.norm(contrib - analytic) / np.linalg.norm(analytic))
    if rel_err > 0.12:
        raise ValueError("Euler contributions off analytic values")
    # VaR contributions should approximately sum to VaR for
    # elliptical distributions (kernel smoothing error bounded).
    var_contrib = euler_var_contributions(x, w, alpha)
    var_total = float(np.quantile(x @ w, alpha))
    var_err = abs(var_contrib.sum() - var_total) / abs(var_total)
    if var_err > 0.15:
        raise ValueError("VaR Euler aggregation loose")
    return {
        "synthetic_euler_es_err": euler_err,
        "synthetic_euler_analytic_rel": rel_err,
        "synthetic_euler_var_err": var_err,
        "synthetic_euler_es_total": es,
        "score": 1.0,
    }
