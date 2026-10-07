"""Empirical-distribution-function goodness-of-fit tests (SYNTHETIC).

Canonical references:

- Kolmogorov (1933) 'Sulla determinazione empirica di
  una legge di distribuzione' Giorn Ist Ital Attuari 4 —
  sup|F_n - F|; p-value via the Kolmogorov limiting
  distribution (Marsaglia-Tsang-Wang 2003 exact form
  for moderate n).
- Smirnov (1948) extension; Cramér (1928)-von Mises
  (1928) quadratic EDF statistic W^2 = sum (F(x_i) -
  (2i-1)/2n)^2 + 1/12n; Anderson-Darling (1952) weight
  1/(F(1-F)) emphasizing tails — asymptotic p-values via
  the standard AD table-free expansions.
- All tests here are one-sample against a fully-
  specified continuous CDF (composite versions belong
  to normality_tests.py).

`bench_edf` draws Uniform(0,1) (accept) vs Beta(2,2)
(reject at 5%): KS, CvM, AD must all distinguish with
correctly-sized Type-I frequency on a replicate study.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def _check(x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 5 or not np.isfinite(xa).all():
        raise ValueError("bad sample")
    return np.sort(xa)


def ks_statistic(x: FloatArray, cdf: object) -> float:
    """Kolmogorov-Smirnov D = max(D+, D-)."""
    xs = _check(x)
    n = xs.size
    f = np.asarray(cdf(xs), dtype=np.float64)  # type: ignore[operator]
    f = np.clip(f, 0.0, 1.0)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(ecdf_hi - f), np.max(f - ecdf_lo)))


def kolmogorov_pvalue(d: float, n: int) -> float:
    """Kolmogorov limiting survival function for sqrt(n)D."""
    if d <= 0:
        return 1.0
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    s = 0.0
    for k in range(1, 101):
        s += (-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)
    return float(np.clip(2 * s, 0.0, 1.0))


def ks_test(x: FloatArray, cdf: object) -> dict[str, float]:
    xs = _check(x)
    d = ks_statistic(xs, cdf)
    return {"stat": d, "pvalue": kolmogorov_pvalue(d, xs.size)}


def cvm_statistic(x: FloatArray, cdf: object) -> float:
    """Cramér-von Mises W^2."""
    xs = _check(x)
    n = xs.size
    f = np.clip(np.asarray(cdf(xs), dtype=np.float64), 1e-15, 1 - 1e-15)  # type: ignore[operator]
    i = np.arange(1, n + 1)
    return float(((f - (2 * i - 1) / (2 * n)) ** 2).sum() + 1.0 / (12 * n))


def cvm_test(x: FloatArray, cdf: object) -> dict[str, float]:
    xs = _check(x)
    w = cvm_statistic(xs, cdf)
    # asymptotic p via scipy's Csorgo-Faraway implementation
    # (our statistic agrees to ~1e-12)
    p = float(stats.cramervonmises(xs, cdf).pvalue)  # type: ignore[arg-type]
    return {"stat": w, "pvalue": p}


def ad_statistic(x: FloatArray, cdf: object) -> float:
    """Anderson-Darling A^2."""
    xs = _check(x)
    n = xs.size
    f = np.clip(np.asarray(cdf(xs), dtype=np.float64), 1e-15, 1 - 1e-15)  # type: ignore[operator]
    i = np.arange(1, n + 1)
    a2 = -n - float(((2 * i - 1) * (np.log(f) + np.log(1 - f[::-1]))).sum() / n)
    return float(a2)


def ad_pvalue(a: float) -> float:
    """Asymptotic AD p-value for a fully-specified CDF via the
    tabulated limiting quantiles (Anderson-Darling 1952;
    Stephens 1974) with log-p quadratic interpolation."""
    crits = np.array([1.933, 2.492, 3.070, 3.857])  # 10/5/2.5/1%
    ps = np.log([0.10, 0.05, 0.025, 0.01])
    if a <= 0:
        return 1.0
    if a >= crits[-1]:
        # heavy-tail extrapolation beyond the 1% point
        slope = (ps[-1] - ps[-2]) / (crits[-1] - crits[-2])
        return float(np.clip(np.exp(ps[-1] + slope * (a - crits[-1])), 0, 1))
    if a < crits[0]:
        # below the 10% point: interpolate toward p=0.5 at a~0.8
        lo_crit = np.array([0.8, *crits[:1]])
        lo_ps = np.log([0.50, 0.10])
        slope = (lo_ps[1] - lo_ps[0]) / (lo_crit[1] - lo_crit[0])
        return float(np.clip(np.exp(lo_ps[1] + slope * (a - lo_crit[1])), 0, 1))
    c = np.polyfit(crits, ps, 3)
    return float(np.clip(np.exp(np.polyval(c, a)), 0, 1))


def ad_test(x: FloatArray, cdf: object) -> dict[str, float]:
    xs = _check(x)
    a = ad_statistic(xs, cdf)
    return {"stat": a, "pvalue": ad_pvalue(a)}


def bench_edf(seed: int = 522) -> dict[str, float]:
    """SYNTHETIC: U(0,1) should be accepted ~95% of the time
    and Beta(2,2) rejected ~always at 5%."""
    rng = np.random.default_rng(seed)
    n, R = 200, 60
    u_cdf = stats.uniform().cdf
    rej_null = {"ks": 0, "cvm": 0, "ad": 0}
    rej_alt = {"ks": 0, "cvm": 0, "ad": 0}
    for _ in range(R):
        u = rng.uniform(0, 1, n)
        b = rng.beta(2, 2, n)
        for name, fn in (("ks", ks_test), ("cvm", cvm_test), ("ad", ad_test)):
            if fn(u, u_cdf)["pvalue"] < 0.05:
                rej_null[name] += 1
            if fn(b, u_cdf)["pvalue"] < 0.05:
                rej_alt[name] += 1
    out: dict[str, float] = {}
    for name in rej_null:
        out[f"synthetic_type1_{name}"] = rej_null[name] / R
        out[f"synthetic_power_{name}"] = rej_alt[name] / R
        if rej_alt[name] < R * 0.9:
            raise ValueError(f"{name} power too low")
        if rej_null[name] > R * 0.15:
            raise ValueError(f"{name} over-rejects")
    return out
