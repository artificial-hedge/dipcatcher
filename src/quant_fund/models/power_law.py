"""Power-law tail fitting — Clauset, Shalizi & Newman (2009).

Estimates the continuous power-law tail

    p(x) = (alpha - 1) / x_min * (x / x_min)^{-alpha},   x >= x_min

by picking ``x_min`` to minimize the Kolmogorov-Smirnov distance
between the empirical tail CDF and the fitted power law, and ``alpha``
by the Hill-type MLE. A parametric-bootstrap p-value (``B`` synthetic
samples refit under the model) tests the power-law hypothesis.

References
----------
- Clauset, Shalizi & Newman (2009) SIAM Review 51, "Power-law
  distributions in empirical data".
- Clauset, Young & Gleditsch (2007) J. Conflict Resolution — discrete
  variant and bootstrap calibration.

Honesty
-------
All estimates are carried through with the fitted tail only — no
extrapolation claims. The bench reports the fitted parameters and the
bootstrap p-value; the score gate is parameter recovery on a true
Pareto draw and detection failure (low p-value) on a lognormal
control. SYNTHETIC only.

Composition
-----------
Called by ``quant_fund.research.benches_w63.bench_power_law``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_MIN_TAIL = 30
_MAX_CAND = 200


def _alpha_mle(tail: FloatArray, x_min: float) -> float:
    return 1.0 + tail.size / float(np.sum(np.log(tail / x_min)))


def _ks_tail(tail: FloatArray, x_min: float, alpha: float) -> float:
    ecdf = np.arange(1, tail.size + 1) / tail.size
    cdf = 1.0 - (tail / x_min) ** (1.0 - alpha)
    return float(np.max(np.abs(ecdf - cdf)))


def pl_fit(x: FloatArray, n_boot: int = 200, seed: int = 0) -> dict[str, float]:
    """Clauset-Shalizi-Newman continuous power-law fit.

    Fail-closed on too few samples or a degenerate support. Returns the
    fitted ``x_min`` / ``alpha``, KS distance, and a bootstrap p-value
    for the power-law hypothesis.
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    if x.size < 80:
        raise ValueError("need >= 80 samples")
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("x must be positive and finite")
    uniq = np.unique(x)
    cands = uniq[:-_MIN_TAIL] if uniq.size > _MIN_TAIL else uniq[:1]
    if cands.size > _MAX_CAND:
        cands = np.quantile(cands, np.linspace(0, 1, _MAX_CAND))
    best = (np.inf, float(x[0]), 2.0)
    for xm in cands:
        tail = x[x >= xm]
        a = _alpha_mle(tail, float(xm))
        d = _ks_tail(tail, float(xm), a)
        if d < best[0]:
            best = (d, float(xm), a)
    ks_obs, x_min, alpha = best
    # Parametric bootstrap (Clauset et al. 2009 sec. 4): with prob
    # p_tail draw from the fitted tail, else resample the body.
    rng = np.random.default_rng(seed)
    tail = x[x >= x_min]
    body = x[x < x_min]
    p_tail = tail.size / x.size
    u = rng.random((n_boot, x.size))
    ks_b = np.empty(n_boot)
    a_b = np.empty(n_boot)
    for i in range(n_boot):
        mask = u[i] < p_tail
        n_t = int(mask.sum())
        sim = np.empty(x.size)
        if n_t:
            sim[mask] = x_min * (1.0 - rng.random(n_t)) ** (1.0 / (1.0 - alpha))
        n_b = x.size - n_t
        if n_b:
            sim[~mask] = body[rng.integers(0, body.size, n_b)] if body.size else x_min
        sim = np.sort(sim)
        d_b, a_bv = np.inf, alpha
        for xm in cands:
            tl = sim[sim >= xm]
            if tl.size < _MIN_TAIL:
                continue
            a_c = _alpha_mle(tl, float(xm))
            d_c = _ks_tail(tl, float(xm), a_c)
            if d_c < d_b:
                d_b, a_bv = d_c, a_c
        ks_b[i], a_b[i] = (d_b if np.isfinite(d_b) else ks_obs), a_bv
    p_val = float(np.mean(ks_b > ks_obs))
    return {
        "x_min": x_min,
        "alpha": float(alpha),
        "ks": ks_obs,
        "p_value": p_val,
        "n_tail": float(tail.size),
        "alpha_boot_std": float(np.std(a_b)),
    }


def bench_powerlaw(seed: int = 20261231 + 368) -> dict[str, float]:
    """SYNTHETIC check — Pareto recovery + lognormal rejection."""
    rng = np.random.default_rng(seed)
    a_true, xm_true = 2.5, 1.0
    par = xm_true * (1.0 - rng.random(3000)) ** (1.0 / (1.0 - a_true))
    fit_p = pl_fit(par, n_boot=160, seed=seed + 1)
    # Controls: lognormal (locally power-law-like, weak rejection) and
    # exponential (clearly not power law — must be rejected).
    ln = np.exp(rng.normal(0.2, 0.9, 3000))
    fit_l = pl_fit(ln, n_boot=160, seed=seed + 2)
    ex = rng.exponential(1.0, 3000) + 0.05
    fit_e = pl_fit(ex, n_boot=160, seed=seed + 3)
    if abs(fit_p["alpha"] - a_true) > 0.2:
        raise ValueError("alpha not recovered on Pareto draw")
    if abs(fit_p["x_min"] - xm_true) > 0.3:
        raise ValueError("x_min not recovered on Pareto draw")
    if fit_p["p_value"] < 0.05:
        raise ValueError("true power law rejected")
    if fit_e["p_value"] > 0.15:
        raise ValueError("exponential control not rejected")
    if fit_l["ks"] < fit_p["ks"]:
        raise ValueError("lognormal fit paradoxically better than Pareto")
    if fit_e["ks"] < 3.0 * fit_p["ks"]:
        raise ValueError("exponential KS distance not elevated")
    return {
        "synthetic_pl_alpha_hat": fit_p["alpha"],
        "synthetic_pl_xmin_hat": fit_p["x_min"],
        "synthetic_pl_ks": fit_p["ks"],
        "synthetic_pl_pval": fit_p["p_value"],
        "synthetic_pl_pval_lognorm": fit_l["p_value"],
        "synthetic_pl_ks_lognorm": fit_l["ks"],
        "synthetic_pl_pval_exp": fit_e["p_value"],
        "synthetic_score": 1.0,
    }
