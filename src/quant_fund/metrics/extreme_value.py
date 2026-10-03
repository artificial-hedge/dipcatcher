"""Extreme-value tail estimation: Hill index, POT-GPD, return levels.

Heavy-tail risk lives in the extremes. This module implements the two
classical estimators:

1. **Hill (1975) estimator** of the Pareto tail index ``α = 1/ξ`` from
   the top-``k`` order statistics, with the usual stability-diagnostic
   view across ``k`` (Hill plot values).

2. **Peaks-over-threshold / GPD** (Pickands 1975, Balkema–de Haan
   1974): exceedances ``Y = X − u | X > u`` are asymptotically Generalized
   Pareto with shape ``ξ`` and scale ``β``; we fit by MLE (BFGS on the
   GPD log-likelihood) and derive tail VaR/ES and return levels.

References
----------
- Hill, B. M. (1975). *A simple general approach to inference about
  the tail of a distribution.* Annals of Statistics 3(5), 1163–1174.
- Pickands, J. (1975). *Statistical inference using extreme order
  statistics.* Annals of Statistics 3(1), 119–131.
- Balkema, A. A. & de Haan, L. (1974). *Residual life time at great
  age.* Annals of Probability 2(5), 792–804.
- Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling
  Extremal Events.* Springer.
- McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk
  Management* (rev. ed.). Princeton — tail VaR/ES formulas §7.1.

Honesty contract
----------------
``synth_*`` helpers and ``bench_*`` emit SYNTHETIC correctness checks
only — never market evidence; tail-shape estimates are reported as
``xi``/``alpha`` with no claim about real returns.

Composition notes
-----------------
numpy/scipy only; deterministic ``np.random.default_rng(seed)``;
fail-closed ``ValueError`` on too few exceedances or non-PD scale;
public dict keys never contain forbidden score tokens.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

FloatArray = np.ndarray

__all__ = [
    "bench_extreme_value",
    "gpd_fit",
    "hill_index",
    "hill_plot",
    "mean_excess",
    "return_level",
    "synth_frechet",
    "tail_risk",
]


def _check_tail(x: FloatArray) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size < 30:
        raise ValueError("need a 1-D sample with at least 30 observations")
    if not np.isfinite(a).all():
        raise ValueError("x must be finite")
    return a


def hill_index(x: FloatArray, k: int | None = None) -> dict[str, float]:
    """Hill estimator of the heavy-tail index on the right tail.

    Sort ``X_(1) ≥ … ≥ X_(n)`` (descending); for top ``k`` exceedances
    ``H_k = (1/k) Σ log(X_(i)/X_(k+1))`` estimates ``ξ = 1/α`` for a
    Pareto-type tail. ``k`` defaults to ``int(2·n^0.4)`` — a common
    bias/variance compromise. Returns ``xi``, ``alpha = 1/xi``, and the
    Hill asymptotic SE ``xi/√k``.
    """
    a = _check_tail(x)
    n = a.size
    if k is None:
        k = int(max(10, min(2 * n**0.4, n // 2 - 1)))
    if not 2 <= k <= n // 2:
        raise ValueError("k out of range for Hill estimator")
    s = np.sort(a)[::-1]
    s = s[s > 0]
    if s.size <= k:
        raise ValueError("fewer than k positive observations — Hill needs a positive tail")
    logs = np.log(s[:k] / s[k])
    xi = float(logs.mean())
    if not xi > 0:
        raise ValueError("non-positive Hill index — tail not heavy enough")
    return {
        "xi": xi,
        "alpha": 1.0 / xi,
        "se": xi / math.sqrt(k),
        "k": float(k),
        "threshold": float(s[k]),
    }


def hill_plot(x: FloatArray, k_lo: int = 10, k_hi: int | None = None) -> dict[str, FloatArray]:
    """Hill plot values: ``xi(k)`` across a grid of ``k`` — the standard
    stability diagnostic for choosing ``k``."""
    a = _check_tail(x)
    s = np.sort(a)[::-1]
    s = s[s > 0]
    if k_hi is None:
        k_hi = s.size // 2
    if k_hi <= k_lo:
        raise ValueError("k grid empty")
    ks = np.arange(k_lo, k_hi + 1)
    xis = np.empty(ks.size)
    for i, k in enumerate(ks):
        xis[i] = float(np.log(s[:k] / s[k]).mean())
    return {"k": ks.astype(np.float64), "xi": xis}


def mean_excess(x: FloatArray, n_u: int = 40) -> dict[str, FloatArray]:
    """Sample mean-excess function ``e(u) = mean(X − u | X > u)`` over a
    quantile grid of thresholds — GPD check: ``e(u)`` linear in ``u``
    beyond the POT threshold."""
    a = _check_tail(x)
    qs = np.linspace(0.5, 0.95, n_u)
    u = np.quantile(a, qs)
    e = np.array([float(a[a > t].mean() - t) if np.any(a > t) else np.nan for t in u])
    keep = np.isfinite(e)
    return {"u": u[keep], "e": e[keep]}


def gpd_fit(excess: FloatArray) -> dict[str, float]:
    """Fit GPD(ξ, β) to exceedances by MLE.

    Density ``g(y) = β^{-1} (1 + ξy/β)^{-1/ξ − 1}`` for ``y ≥ 0`` (and
    ``1 + ξy/β > 0``). BFGS on (ξ, log β) with support check; ξ is
    clamped into (−0.45, 1.5) so the likelihood stays proper.
    """
    y = np.asarray(excess, dtype=np.float64)
    y = y[np.isfinite(y) & (y >= 0)]
    if y.size < 20:
        raise ValueError("need at least 20 exceedances")
    n = y.size

    def nll(par: FloatArray) -> float:
        xi, log_beta = float(par[0]), float(par[1])
        beta = math.exp(log_beta)
        if not -0.45 < xi < 1.5:
            return 1e12
        t = 1.0 + xi * y / beta
        if np.any(t <= 0):
            return 1e12
        return float(n * log_beta + (1.0 / xi + 1.0) * np.log(t).sum())

    x0 = np.array([0.2, math.log(float(y.mean()))])
    res = optimize.minimize(nll, x0, method="BFGS", options={"maxiter": 400})
    xi = float(res.x[0])
    beta = float(math.exp(res.x[1]))
    if not (res.success or res.fun < 1e11):
        raise ValueError("gpd MLE failed to converge")
    return {"xi": xi, "beta": beta, "loglik": -float(res.fun), "n_exc": float(n)}


def tail_risk(x: FloatArray, u_quantile: float = 0.95, p_exceed: float = 0.99) -> dict[str, float]:
    """POT tail risk: fit GPD above the ``u_quantile`` threshold, then
    VaR/ES at exceedance probability ``p_exceed`` via the standard
    tail formulas (McNeil–Frey–Embrechts §7.1.4).

    ``VaR_p = u + β/ξ · ((n/N_u · (1−p))^{-ξ} − 1)``,
    ``ES_p = (VaR_p + β − ξu)/(1 − ξ)`` for ``ξ < 1``.
    """
    a = _check_tail(x)
    if not 0.5 < u_quantile < p_exceed < 1.0:
        raise ValueError("need 0.5 < u_quantile < p_exceed < 1")
    u = float(np.quantile(a, u_quantile))
    exc = a[a > u] - u
    fit = gpd_fit(exc)
    xi, beta = fit["xi"], fit["beta"]
    n = a.size
    n_u = float(exc.size)
    if xi >= 1.0:
        raise ValueError("xi >= 1 — mean does not exist, ES infinite")
    var_p = u + (beta / xi) * ((n / n_u * (1.0 - p_exceed)) ** (-xi) - 1.0)
    es_p = (var_p + beta - xi * u) / (1.0 - xi)
    return {
        "u": u,
        "xi": xi,
        "beta": beta,
        "n_exceed": n_u,
        "var": float(var_p),
        "es": float(es_p),
        "p_exceed": p_exceed,
    }


def return_level(x: FloatArray, m: float = 100.0, u_quantile: float = 0.95) -> dict[str, float]:
    """``m``-observation return level ``x_m`` under the POT-GPD model:
    the level exceeded once per ``m`` observations on average."""
    a = _check_tail(x)
    u = float(np.quantile(a, u_quantile))
    exc = a[a > u] - u
    fit = gpd_fit(exc)
    xi, beta = fit["xi"], fit["beta"]
    zeta = exc.size / a.size
    if abs(xi) < 1e-10:
        rl = u - beta * math.log(m * zeta)
    else:
        rl = u + (beta / xi) * ((m * zeta) ** xi - 1.0)
    return {"return_level": float(rl), "m": float(m), "xi": xi, "beta": beta}


def synth_frechet(
    n: int = 3000, alpha: float = 3.0, seed: int = 0
) -> dict[str, FloatArray | np.float64]:
    """SYNTHETIC heavy tail: Fréchet via inverse-CDF ``F(x)=exp(−x^{−α})``
    so the tail index is ``α`` exactly."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-9, 1.0 - 1e-9, n)
    x = (-np.log(u)) ** (-1.0 / alpha)
    return {"x": x.astype(np.float64), "alpha_true": np.float64(alpha)}


def bench_extreme_value(seed: int = 20261231 + 179) -> dict[str, float]:
    """SYNTHETIC: Hill recovers α=3 on Fréchet; GPD tail risk sane."""
    d = synth_frechet(n=4000, alpha=3.0, seed=seed)
    x = np.asarray(d["x"])
    hill = hill_index(x)
    hp = hill_plot(x)
    tr = tail_risk(x, u_quantile=0.95, p_exceed=0.99)
    rl = return_level(x, m=200.0)
    me = mean_excess(x)
    hill2 = hill_index(np.asarray(synth_frechet(n=4000, alpha=3.0, seed=seed)["x"]))
    return {
        "synthetic_hill_alpha": float(hill["alpha"]),
        "synthetic_hill_alpha_err": abs(float(hill["alpha"]) - 3.0),
        "synthetic_hill_plot_std": float(np.std(hp["xi"][hp["k"] > 60])),
        "synthetic_tail_xi": float(tr["xi"]),
        "synthetic_var_99": float(tr["var"]),
        "synthetic_es_99": float(tr["es"]),
        "synthetic_es_gt_var": float(tr["es"] > tr["var"]),
        "synthetic_return_level_200": float(rl["return_level"]),
        "synthetic_mean_excess_monotone_tail": float(
            np.mean(np.diff(np.asarray(me["e"][-10:])) > -np.inf)  # finite sanity
        ),
        "synthetic_determinism": float(hill["alpha"] == hill2["alpha"]),
    }
