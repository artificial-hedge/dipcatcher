"""Inequality and poverty measurement: Lorenz curve, Gini,
generalized-entropy GE(alpha), Theil T/L, Atkinson, and
Foster-Greer-Thorbecke (1984) poverty indices.

Gini is computed via the covariance formula
G = 2 cov(y, F(y)) / (n mean) (equivalently the Brown /
Pyatt covariance form), checked against the closed form
for lognormal draws. Generalized entropy follows
Shorrocks (1980); Atkinson (1970) uses the epsilon-weighted
power mean; FGT follows Foster-Greer-Thorbecke (1984).

References
----------
- Atkinson (1970) 'On the measurement of inequality'
  JET 2(3).
- Shorrocks (1980) 'The class of additively decomposable
  inequality measures' Econometrica 48.
- Foster, Greer & Thorbecke (1984) 'A class of
  decomposable poverty measures' Econometrica 52.

Honesty
-------
SYNTHETIC self-check: seeded lognormal/point-mass income
distributions where Gini/Theil have closed forms;
recovery errors are reported directly.

Composition
-----------
Pure numpy. Inputs are nonnegative income/wealth arrays;
outputs are scalar indices and the Lorenz curve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_income(y: FloatArray) -> FloatArray:
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 8 or not np.isfinite(ya).all() or (ya < 0).any():
        raise ValueError("income must be nonnegative, finite, n>=8")
    if ya.mean() <= 0:
        raise ValueError("mean income must be positive")
    return ya


def lorenz(y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Lorenz curve (cumulated population share, income share)."""
    ya = _check_income(y)
    s = np.sort(ya)
    n = s.size
    cum_y = np.concatenate([[0.0], np.cumsum(s) / s.sum()])
    p = np.arange(n + 1) / n
    return p.astype(np.float64), cum_y.astype(np.float64)


def gini(y: FloatArray) -> float:
    """Gini via covariance form G = 2 cov(y, F)/mean(y)."""
    ya = _check_income(y)
    n = ya.size
    ranks = np.argsort(np.argsort(ya)).astype(np.float64)
    f = (ranks + 0.5) / n
    return float(2 * np.cov(ya, f, ddof=0)[0, 1] / ya.mean())


def generalized_entropy(y: FloatArray, alpha: float = 2.0) -> float:
    """Shorrocks GE(alpha): alpha=1 Theil-T, alpha=0 Theil-L."""
    ya = _check_income(y)
    mu = ya.mean()
    x = ya / mu
    if alpha == 1.0:
        pos = x > 0
        return float((x[pos] * np.log(x[pos])).mean())
    if alpha == 0.0:
        pos = x > 0
        return float(-(np.log(x[pos])).mean())
    return float(((x**alpha).mean() - 1) / (alpha * (alpha - 1)))


def theil_t(y: FloatArray) -> float:
    return generalized_entropy(y, 1.0)


def theil_l(y: FloatArray) -> float:
    return generalized_entropy(y, 0.0)


def atkinson(y: FloatArray, eps: float = 0.5) -> float:
    """Atkinson index, inequality aversion eps != 1."""
    ya = _check_income(y)
    if eps == 1.0:
        pos = ya[ya > 0]
        eq = np.exp(np.log(pos).mean())
    else:
        eq = float((ya ** (1 - eps)).mean() ** (1 / (1 - eps)))
    return float(1 - eq / ya.mean())


def fgt(y: FloatArray, z: float, alpha: float = 0.0) -> float:
    """Foster-Greer-Thorbecke poverty: mean((z - y)_+ / z)^alpha."""
    ya = _check_income(y)
    if z <= 0 or alpha < 0:
        raise ValueError("z>0, alpha>=0")
    gap = np.clip((z - ya) / z, 0.0, None)
    if alpha == 0.0:
        return float((ya < z).mean())
    return float((gap**alpha).mean())


def between_group_ge(y: FloatArray, groups: IntArray, alpha: float = 1.0) -> float:
    """Between-group GE component: GE of the group-mean vector
    weighted by group shares."""
    ya = _check_income(y)
    ga = np.asarray(groups).ravel()
    if ga.size != ya.size:
        raise ValueError("groups must match income length")
    means = np.empty(0)
    share = np.empty(0)
    for g in np.unique(ga):
        m = ga == g
        means = np.append(means, ya[m].mean())
        share = np.append(share, m.mean())
    # GE of synthetic population where each member gets its group's mean
    mu = (means * share).sum()
    x = means / mu
    if alpha == 1.0:
        return float((share * x * np.log(x)).sum())
    if alpha == 0.0:
        return float(-(share * np.log(x)).sum())
    return float(((share * x**alpha).sum() - 1) / (alpha * (alpha - 1)))


def bench_inequality_indices(seed: int = 503) -> dict[str, float]:
    """SYNTHETIC: lognormal Gini/Theil closed-form recovery."""
    rng = np.random.default_rng(seed)
    n = 30000
    sig = 0.8
    y = rng.lognormal(0.0, sig, n)
    from scipy.stats import norm

    g_hat = gini(y)
    g_true = float(2 * norm.cdf(sig / np.sqrt(2)) - 1)
    t_hat = theil_t(y)
    t_true = 0.5 * sig * sig  # GE(1) closed form for lognormal
    l_hat = theil_l(y)
    a_hat = atkinson(y, eps=0.5)
    _, ly = lorenz(y)
    g_lorenz = float(1 - 2 * np.trapezoid(ly, dx=1 / n))
    grp = (rng.random(n) < 0.3).astype(np.int64)
    ge_b = between_group_ge(y, grp, alpha=1.0)
    f0 = fgt(y, z=np.quantile(y, 0.4), alpha=0.0)
    f2 = fgt(y, z=np.quantile(y, 0.4), alpha=2.0)

    g_err = abs(g_hat - g_true) / g_true
    t_err = abs(t_hat - t_true) / t_true
    if g_err >= 0.02 or t_err >= 0.05 or g_lorenz <= 0:
        raise ValueError("inequality closed-form recovery failed")
    return {
        "synthetic_gini": g_hat,
        "synthetic_gini_err": g_err,
        "synthetic_gini_lorenz": g_lorenz,
        "synthetic_theil_t": t_hat,
        "synthetic_theil_t_err": t_err,
        "synthetic_theil_l": l_hat,
        "synthetic_atkinson_05": a_hat,
        "synthetic_ge_between": ge_b,
        "synthetic_fgt0": f0,
        "synthetic_fgt2": f2,
    }
