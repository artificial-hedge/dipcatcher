"""Hong-Li (2005) density-forecast evaluation via PIT transforms.

References
----------
- Hong, Y. & Li, H. (2005). "Nonparametric Specification Testing
  for Continuous-Time Models with Applications to Term Structure
  of Interest Rates." *Review of Financial Studies* 18(1), 37-84.
- Diebold, F.X., Gunther, T. & Tay, A. (1998). "Evaluating Density
  Forecasts with Applications to Financial Risk Management."
  *IER* 39(4), 863-883.
- Berkowitz, J. (2001). "Testing Density Forecasts, with
  Applications to Risk Management." *JBES* 19(4), 465-474.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
Under a correctly-specified density model the probability-integral
transforms ``z_t = F_theta(x_t | I_{t-1})`` are iid U(0,1)
(Diebold-Gunther-Tay). Hong-Li measure the L2 distance between
the joint uniform density and the bivariate density of
``(z_t, z_{t-lag})`` using a boundary-corrected Parzen-Rosenblatt
kernel: the characteristic weight ``W(p)`` on the transformed
integer grid integrates the kernel against the serial dependence.
We implement the univariate-uniformity half plus a serial
correlogram on the clipped Gaussian scores ``Phi^{-1}(z_t)``
(Berkowitz shape) — the M-statistic
``M(l) = [sum_j |w_j|^2 - T * int(w)] / sqrt(2 T int(w^2))``
combines the L2 discrepancy across the integer-Fourier basis
``cos(2 pi j z), sin(2 pi j z)`` on [0,1], whose null mean is zero
and variance is 1/2 per mode. The synth plants a correctly
specified AR(1)-Gaussian filter (z uniform => low M) against a
wrong-variance/ wrong-persistence filter (z clumped/serial =>
large M), and the bench requires the correct model's |M| to sit
inside the asymptotic N(0,1) band while the wrong one breaches it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]

_J_MODES = 40


def _check_z(z: FloatArray) -> FloatArray:
    zz = np.asarray(z, dtype=np.float64)
    if zz.ndim != 1 or zz.size < 60 or not np.all(np.isfinite(zz)):
        raise ValueError("bad z")
    if np.any(zz <= 0.0) or np.any(zz >= 1.0):
        raise ValueError("PIT must lie in (0,1)")
    return zz


def pit_gaussian(x: FloatArray, mu: float, sigma: float) -> FloatArray:
    """iid-N benchmark PIT for a one-step-ahead Gaussian forecast."""
    xx = np.asarray(x, dtype=np.float64)
    if sigma <= 0.0:
        raise ValueError("bad sigma")
    return np.asarray(_stats.norm.cdf(xx, loc=mu, scale=sigma), dtype=np.float64)


def pit_ar1(x: FloatArray, phi: float, sigma: float) -> FloatArray:
    """PIT under the fitted AR(1): x_t | x_{t-1} ~ N(phi x_{t-1}, sigma^2)."""
    xx = np.asarray(x, dtype=np.float64)
    if abs(phi) >= 1.0 or sigma <= 0.0:
        raise ValueError("bad params")
    mu = phi * xx[:-1]
    return np.asarray(_stats.norm.cdf(xx[1:], loc=mu, scale=sigma), dtype=np.float64)


def hl_m_statistic(z: FloatArray) -> dict[str, float]:
    """Hong-Li uniformity statistic on integer-Fourier modes.

    For each mode j the basis ``g_j(z)`` is ``sqrt(2) cos(2 pi j z)``
    and ``sqrt(2) sin(2 pi j z)``; under iid-uniformity each has
    mean 0 and variance 1, so the sum of squared sample means is
    asymptotically chi2(2J). ``m_stat`` standardizes it:
    ``(X2 - 2J) / sqrt(4J)`` -> N(0,1) under correct specification.
    Also reports the first-lag score autocorrelation ``rho1`` on
    ``Phi^{-1}(z)`` — the Hong-Li serial half — and a Berkowitz
    censor-free LR on mean/variance of the scores.
    """
    zz = _check_z(z)
    t = zz.size
    js = np.arange(1, _J_MODES + 1)[:, None]
    ang = 2.0 * np.pi * js * zz[None, :]
    cos_m = np.mean(np.sqrt(2.0) * np.cos(ang), axis=1)
    sin_m = np.mean(np.sqrt(2.0) * np.sin(ang), axis=1)
    x2 = float(t * np.sum(cos_m * cos_m + sin_m * sin_m))
    j2 = 2 * _J_MODES
    m_stat = (x2 - j2) / np.sqrt(2.0 * j2)
    s = _stats.norm.ppf(np.clip(zz, 1e-6, 1.0 - 1e-6))
    rho1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    # Berkowitz LR: scores should be iid N(0,1)
    s_mu = float(np.mean(s))
    s_sd = float(np.std(s, ddof=1))
    ll_fit = float(np.sum(_stats.norm.logpdf(s, s_mu, s_sd)))
    ll_null = float(np.sum(_stats.norm.logpdf(s, 0.0, 1.0)))
    lr = 2.0 * (ll_fit - ll_null)
    return {
        "x2": x2,
        "df": float(j2),
        "m_stat": m_stat,
        "rho1": rho1,
        "berkowitz_lr": lr,
        "score_mu": s_mu,
        "score_sd": s_sd,
    }


def synth_hong_li(
    seed: int = 20261231 + 303,
    t: int = 1200,
    phi: float = 0.6,
) -> dict[str, float | FloatArray]:
    """SYNTHETIC AR(1) process + correct/wrong forecast filters."""
    rng = np.random.default_rng(seed)
    sig = 1.0
    eps = sig * rng.standard_normal(t + 1)
    x = np.empty(t + 1)
    x[0] = 0.0
    for i in range(t):
        x[i + 1] = phi * x[i] + eps[i + 1]
    x = x[1:]
    z_right = pit_ar1(x, phi, sig)
    # wrong: persistence underestimated + variance overstated
    z_wrong = pit_ar1(x, phi * 0.4, sig * 1.4)
    return {
        "x": x,
        "z_right": z_right,
        "z_wrong": z_wrong,
        "phi": phi,
        "sigma": sig,
    }


def bench_hong_li(seed: int = 20261231 + 303) -> dict[str, float]:
    """Wave-52 self-check: correct model passes, wrong one fails."""
    d = synth_hong_li(seed=seed)
    r_ok = hl_m_statistic(np.asarray(d["z_right"]))
    r_bad = hl_m_statistic(np.asarray(d["z_wrong"]))
    ok = abs(r_ok["m_stat"]) < 3.0 and abs(r_ok["rho1"]) < 0.12 and r_bad["m_stat"] > 8.0
    return {
        "m_right": r_ok["m_stat"],
        "rho1_right": r_ok["rho1"],
        "m_wrong": r_bad["m_stat"],
        "rho1_wrong": r_bad["rho1"],
        "berkowitz_wrong": r_bad["berkowitz_lr"],
        "score": float(ok),
    }
