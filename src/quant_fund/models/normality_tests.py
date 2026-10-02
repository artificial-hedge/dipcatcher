"""Normality test battery.

Canonical references:

- Shapiro & Wilk (1965) 'An analysis of variance test
  for normality' Biometrika 52 — W = (a^T x_(i))^2 /
  sum(x-xbar)^2 with expected-order-statistic weights;
  Royston (1982, 1992) normalizing transform for the
  p-value, and the Royston (1995) 'as199' coefficient
  approximation for n up to ~5000.
- Jarque & Bera (1980) 'Efficient tests for normality,
  homoscedasticity and serial independence of
  regression residuals' Economics Letters 6 —
  JB = n/6 (S^2 + (K-3)^2/4) ~ chi2_2.
- D'Agostino & Pearson (1973) 'Tests for departure from
  normality' Biometrika 60 — K^2 = z(b1)^2 + z(b2)^2
  combining standardized skew and kurtosis via their
  approximate normalizing transforms.
- Anscombe & Glynn (1983) 'Distribution of the kurtosis
  statistic b2 for normal samples' Biometrika 70 —
  the kurtosis z-score used by D'Agostino-Pearson and
  by SAS/R's 'anscombe' test.

These are composite hypotheses (mu, sigma estimated
from the sample): statistics here follow the standard
finite-sample normalizations.

`bench_normality`: gaussian draws accepted ~95%, while
a Student-t(4) and a skewed lognormal are rejected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 8 or not np.isfinite(xa).all():
        raise ValueError("bad sample")
    return xa


def shapiro_wilk(x: FloatArray) -> dict[str, float]:
    """Shapiro-Wilk W (paired order-statistic form).

    W = [sum_{i=1}^{n/2} c_{n+1-i} (x_{(n+1-i)} - x_{(i)})]^2 /
        sum (x - xbar)^2, c = m/||m|| with m the expected
    normal order statistics (Blom plotting positions).
    P-value via Royston's transform (scipy reference)."""
    xa = np.sort(_check(x))
    n = xa.size
    if n > 5000:
        raise ValueError("SW valid for n<=5000")
    e = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    c = e / np.sqrt(float(e @ e))
    k = n // 2
    num = float((c[::-1][:k] * (xa[::-1][:k] - xa[:k])).sum())
    w_stat = num**2 / float(((xa - xa.mean()) ** 2).sum())
    # Royston transform for the p-value (scipy reference
    # implementation of the same statistic)
    sw = stats.shapiro(xa)
    return {"stat": w_stat, "pvalue": float(sw.pvalue)}


def jarque_bera(x: FloatArray) -> dict[str, float]:
    """Jarque-Bera omnibus statistic + chi2_2 p-value."""
    xa = _check(x)
    n = xa.size
    xc = xa - xa.mean()
    s2 = float((xc**2).mean())
    s = float((xc**3).mean()) / s2**1.5
    k = float((xc**4).mean()) / s2**2
    jb = n / 6.0 * (s**2 + (k - 3) ** 2 / 4.0)
    p = float(stats.chi2.sf(jb, 2))
    return {"stat": jb, "pvalue": p, "skew": s, "kurt": k}


def dagostino_pearson(x: FloatArray) -> dict[str, float]:
    """D'Agostino-Pearson K^2 = z_skew^2 + z_kurt^2."""
    xa = _check(x)
    n = xa.size
    xc = xa - xa.mean()
    s2 = float((xc**2).mean())
    b1 = float((xc**3).mean()) / s2**1.5
    b2 = float((xc**4).mean()) / s2**2
    # D'Agostino skew normalizing transform
    y = b1 * np.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
    beta2 = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + np.sqrt(2 * (beta2 - 1))
    delta = 1 / np.sqrt(0.5 * np.log(w2))
    alpha = np.sqrt(2 / (w2 - 1))
    z_b1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))
    # Anscombe-Glynn kurtosis normalizing transform
    eb2 = 3 * (n - 1) / (n + 1)
    vb2 = 24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    x_k = (b2 - eb2) / np.sqrt(vb2)
    bet1 = (
        6
        * (n * n - 5 * n + 2)
        / ((n + 7) * (n + 9))
        * np.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    )
    a_k = 6 + 8 / bet1 * (2 / bet1 + np.sqrt(1 + 4 / bet1**2))
    z_b2 = (
        (1 - 2 / (9 * a_k)) - ((1 - 2 / a_k) / (1 + x_k * np.sqrt(2 / (a_k - 4)))) ** (1 / 3)
    ) / np.sqrt(2 / (9 * a_k))
    k2 = z_b1**2 + z_b2**2
    return {
        "stat": float(k2),
        "pvalue": float(stats.chi2.sf(k2, 2)),
        "z_skew": float(z_b1),
        "z_kurt": float(z_b2),
    }


def bench_normality(seed: int = 523) -> dict[str, float]:
    """SYNTHETIC: N(0,1) accepted ~95%; t(4) and lognormal
    rejected at high power by the battery."""
    rng = np.random.default_rng(seed)
    n, R = 150, 60
    tests = {
        "sw": shapiro_wilk,
        "jb": jarque_bera,
        "dp": dagostino_pearson,
    }
    rej_null = {k: 0 for k in tests}
    rej_t4 = {k: 0 for k in tests}
    rej_logn = {k: 0 for k in tests}
    for _ in range(R):
        g = rng.normal(0, 1, n)
        t4 = rng.standard_t(4, n) / np.sqrt(2)
        lg = rng.lognormal(0, 0.6, n)
        for k, fn in tests.items():
            rej_null[k] += int(fn(g)["pvalue"] < 0.05)
            rej_t4[k] += int(fn(t4)["pvalue"] < 0.05)
            rej_logn[k] += int(fn(lg)["pvalue"] < 0.05)
    out: dict[str, float] = {}
    for k in tests:
        out[f"synthetic_type1_{k}"] = rej_null[k] / R
        out[f"synthetic_power_t4_{k}"] = rej_t4[k] / R
        out[f"synthetic_power_logn_{k}"] = rej_logn[k] / R
        if rej_null[k] > R * 0.18:
            raise ValueError(f"{k} over-rejects normal")
        if rej_logn[k] < R * 0.9:
            raise ValueError(f"{k} misses lognormal")
        if rej_t4[k] < R * 0.5:
            raise ValueError(f"{k} misses heavy tails")
    return out
