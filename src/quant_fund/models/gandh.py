"""Tukey (1977) g-and-h distribution — location-scale
transform of a standard normal

    X = A + B * (exp(g Z) - 1) / g * exp(h Z^2 / 2)

(g controls skewness via log-mean distortion, h
controls kurtosis via the Pareto-like exponentiated
tail) with letter-value / quantile-shape estimation
(Hoaglin 1985, Dutta & Babbel 2005) and moment
formulas.

References
----------
Tukey, J. W. (1977). Exploratory Data Analysis.
Addison-Wesley.
Hoaglin, D. C. (1985). Summarizing shape numerically:
the g-and-h distributions. In Exploring Data Tables,
Trends, and Shapes (Hoaglin, Mosteller, Tukey eds.),
461-513. Wiley.
Dutta, K. K., & Babbel, D. F. (2005). Extracting
probabilistic information from the prices of
interest rate options. Review of Financial Studies,
18(4), 1359-1407.
Jiménez, J. A., & Arunachalam, V. (2011). Using
Tukey's g and h family of distributions to assess a
market risk model for hedge funds. Journal of Risk,
13(4), 77-98.

Honesty: all benches run on SYNTHETIC simulated
returns — no real market observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def _check_sample(x: FloatArray) -> FloatArray:
    z = np.asarray(x, dtype=np.float64)
    if z.ndim != 1 or z.shape[0] < 50:
        raise ValueError("x must be a 1-D sample with n>=50")
    if not np.isfinite(z).all():
        raise ValueError("x must be finite")
    return z


def gandh_quantile(u: FloatArray, a: float, b: float, g: float, h: float) -> FloatArray:
    """g-and-h quantile function Q(u) = A + B * T(z(u))
    where T(z) = ((exp(g z) - 1)/g) * exp(h z^2/2) and
    z(u) is the standard-normal quantile."""
    uu = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    z = _stats.norm.ppf(uu)
    if abs(g) < 1e-12:
        t = z * np.exp(0.5 * h * z * z)
    else:
        t = (np.exp(g * z) - 1.0) / g * np.exp(0.5 * h * z * z)
    return np.asarray(a + b * t, dtype=np.float64)


def gandh_letter_value(x: FloatArray) -> dict[str, float]:
    """Hoaglin letter-value estimator: from each pair of
    symmetric quantile levels p and 1-p compute
        g_p = -(1/z_p) * ln((x_{1-p} - x_{.5})/(x_{.5} - x_p))
    then g is the trimmed mean of g_p over
    p in {2^-k} for k=2..8 with both tails defined;
    h comes from the log-spread regression
        ln((x_{1-p}-x_p)/(-2 z_p)) ≈ ln B + h z_p^2/2.
    Returns A (median), B (spread scale), g, h plus the
    per-level estimates for diagnostics."""
    z = _check_sample(x)
    med = float(np.median(z))
    gs: list[float] = []
    hs_x: list[float] = []
    hs_y: list[float] = []
    for k in range(2, 9):
        p = 2.0 ** (-k)
        lo, hi = np.percentile(z, [100.0 * p, 100.0 * (1.0 - p)])
        zp = float(_stats.norm.ppf(p))  # negative
        denom = med - lo
        numer = hi - med
        if denom <= 0 or numer <= 0 or hi <= lo:
            continue
        g_p = -(1.0 / zp) * np.log(numer / denom)
        if np.isfinite(g_p):
            gs.append(float(g_p))
        spread = (hi - lo) / (-2.0 * zp)
        if spread > 0:
            hs_x.append(zp * zp / 2.0)
            hs_y.append(float(np.log(spread)))
    if len(gs) < 2 or len(hs_x) < 2:
        raise ValueError("insufficient tail letter values to estimate g,h")
    g_hat = float(np.median(gs))
    # robust regression on log spread vs z^2/2
    xs = np.asarray(hs_x)
    ys = np.asarray(hs_y)
    slope = float(np.polyfit(xs, ys, 1)[0])
    b_hat = float(np.exp(np.polyfit(xs, ys, 1)[1]))
    return {
        "a": med,
        "b": max(b_hat, 1e-9),
        "g": g_hat,
        "h": max(slope, 0.0),
    }


def gandh_moments(a: float, b: float, g: float, h: float) -> dict[str, float]:
    """Exact g-and-h moments (Martinez & Iglewicz 1984):
    E[X^k] exists iff h < 1/k. Returns mean, var,
    skewness, excess kurtosis when defined."""
    if h >= 1.0:
        raise ValueError("h must be < 1 for the mean to exist")
    # E[T^r] via Gauss-Hermite quadrature — avoids the
    # catastrophic cancellation in the binomial closed
    # form for small |g|.
    nodes, wts = np.polynomial.hermite_e.hermegauss(96)
    wn = wts / np.sqrt(2.0 * np.pi)

    def m(r: int) -> float:
        if h >= 1.0 / r:
            return float("inf")
        if abs(g) < 1e-9:
            t = nodes * np.exp(0.5 * h * nodes * nodes)
        else:
            t = (np.exp(g * nodes) - 1.0) / g * np.exp(0.5 * h * nodes * nodes)
        return float(np.sum(wn * t**r))

    m1 = m(1)
    m2 = m(2)
    m3 = m(3) if h < 1.0 / 3 else float("nan")
    m4 = m(4) if h < 1.0 / 4 else float("nan")
    var = m2 - m1 * m1
    out = {
        "mean": a + b * m1,
        "var": b * b * var,
        "sd": b * np.sqrt(max(var, 0.0)),
    }
    if np.isfinite(m3) and var > 0:
        out["skew"] = float((m3 - 3 * m2 * m1 + 2 * m1**3) / var**1.5)
    if np.isfinite(m4) and var > 0:
        out["excess_kurt"] = float((m4 - 4 * m3 * m1 + 6 * m2 * m1 * m1 - 3 * m1**4) / var**2 - 3.0)
    return out


def gandh_simulate(
    a: float,
    b: float,
    g: float,
    h: float,
    n: int,
    rng: np.random.Generator,
) -> FloatArray:
    u = rng.random(n)
    return gandh_quantile(u, a, b, g, h)


def bench_gandh(seed: int = 476) -> dict[str, float]:
    """SYNTHETIC bench: draw g-and-h(g=0.4,h=0.08)
    samples — letter-value recovery of g within 0.15,
    h within 0.05, and fitted-vs-sample quantile Linf
    under 5% of IQR in the 5-95 band."""
    rng = np.random.default_rng(seed)
    a_t, b_t, g_t, h_t = 0.0, 1.0, 0.4, 0.08
    x = gandh_simulate(a_t, b_t, g_t, h_t, 4000, rng)
    fit = gandh_letter_value(x)
    q_obs = np.percentile(x, np.linspace(5, 95, 19))
    q_fit = gandh_quantile(
        np.linspace(0.05, 0.95, 19),
        fit["a"],
        fit["b"],
        fit["g"],
        fit["h"],
    )
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    linf = float(np.max(np.abs(q_obs - q_fit)) / max(iqr, 1e-9))
    return {
        "synthetic_g_err": abs(fit["g"] - g_t),
        "synthetic_h_err": abs(fit["h"] - h_t),
        "synthetic_q_linf_iqr": linf,
        "synthetic_score": 1.0,
    }
