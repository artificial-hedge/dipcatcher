"""Fractional Ornstein-Uhlenbeck: spectral simulation + Whittle estimation.

The stationary fOU solves ``dX_t = -a X_t dt + sigma dB^H_t``; its spectral
density is the fBM spectrum filtered by the OU transfer function,

    f(w) = sigma^2 * C_H * |w|^{1 - 2H} / (a^2 + w^2),   w != 0,

with ``C_H = sin(pi H) Gamma(2H + 1) / (2 pi)`` (H=1/2 recovers the
standard OU Lorentzian).  This module simulates the process by circulant
spectral embedding (exact in the Gaussian-stationary sense, up to grid
truncation) and estimates ``(H, a, sigma)`` by Whittle likelihood —
complementing ``models/rough_vol.py``, which provides moment-scaling Hurst
diagnostics and an approximate path kernel simulation (not joint
estimation).

References
----------
- Cheridito, Kawaguchi & Maejima (2003). Fractional Ornstein-Uhlenbeck
  processes. *Electron. J. Probab.* 8 — arXiv:math/0301101 (verified;
  fOU definition and covariance structure).
- Brouste & Iacus (2013). Parameter estimation for the discretely
  observed fOU. *Stat. Probab. Lett.* 83 — Whittle/quadratic-form
  estimation approach (journal).
- Whittle (1951). Hypothesis testing in time series analysis (journal;
  the spectral-domain likelihood used here).
- Fukasawa (2017) / Bayer, Friz & Gatheral (2016) — rough-volatility
  context; H < 1/2 regime (arXiv:1410.3394 verified elsewhere in repo).

Honesty
-------
All benches run on seeded SYNTHETIC fOU paths generated in-module.
Recovery numbers validate the estimation machinery only — never market
evidence.

Composition notes
-----------------
- ``models/rough_vol.py``: moment-scaling ``logvol_hurst``,
  ``variance_curve_fit``, approximate fOU path sim. This module is the
  spectral/exact-likelihood complement (H=1/2 limit, joint (H,a,sigma)).
- ``models/multifractal_vol.py``: fGn synthesis helper and structure
  functions — sibling spectral-simulation consumer.
- ``models/rough_heston_rbergomi.py``: option pricing under rough vol —
  downstream consumer of H estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

FloatArray = NDArray[np.float64]


def fou_spectrum(omega: FloatArray, h: float, a: float, sigma: float = 1.0) -> FloatArray:
    """fOU spectral density f(w) = sigma^2 C_H |w|^{1-2H} / (a^2 + w^2).

    ``omega`` may include 0 (value at 0 is finite for H>1/2, sigma^2
    C_H * 0^{1-2H}/a^2 handled via limit — for H < 1/2 it is +inf and we
    clip to the smallest nonzero frequency's value).
    """
    w = np.asarray(omega, dtype=float)
    if not 0.0 < h < 1.0 or a < 0.0 or sigma <= 0.0:
        raise ValueError("need 0<H<1, a>=0, sigma>0")
    c_h = np.sin(np.pi * h) * math.gamma(2 * h + 1) / (2 * np.pi)
    wabs = np.abs(w)
    nz = wabs > 0
    spec = np.empty(w.size)
    spec[nz] = sigma**2 * c_h * wabs[nz] ** (1 - 2 * h) / (a**2 + w[nz] ** 2)
    if np.any(~nz):
        spec[~nz] = spec[nz].min() if nz.any() else 0.0
    return spec


def simulate_fou_exact(
    n: int,
    h: float,
    a: float,
    sigma: float = 1.0,
    seed: int = 0,
) -> FloatArray:
    """Stationary fOU via circulant spectral embedding.

    Draws complex Gaussian Fourier coefficients with amplitudes set by
    the spectrum and inverse-FFTs — the result is (approximately) a
    stationary Gaussian process with the fOU spectrum. Unit-mean-free:
    returned path is de-meaned.
    """
    if n < 32:
        raise ValueError("n >= 32 required")
    if not 0.0 < h < 1.0 or a < 0.0 or sigma <= 0.0:
        raise ValueError("need 0<H<1, a>=0, sigma>0")
    rng = np.random.default_rng(seed)
    m = 1 << int(np.ceil(np.log2(2 * n)))  # padded grid
    freqs = np.fft.fftfreq(m) * 2 * np.pi  # angular frequencies
    spec = fou_spectrum(np.abs(freqs), h, a, sigma)
    amp = np.sqrt(np.maximum(spec, 0.0) * m / 2.0)
    # complex normal coefficients with conjugate symmetry
    re = rng.standard_normal(m)
    im = rng.standard_normal(m)
    coeff = amp * (re + 1j * im)
    coeff[0] = 0.0  # zero DC — de-meaned process
    x = np.fft.ifft(coeff).real[:n]
    x = x - x.mean()
    scale = np.std(x)
    return np.asarray(x / scale * sigma) if scale > 0 else x


def whittle_loglik(x: FloatArray, h: float, a: float, sigma: float) -> float:
    """Whittle pseudo-likelihood of a stationary series under fOU spectrum.

    ``-2 L ~ sum_k [ log f(w_k) + I(w_k) / f(w_k) ]`` over positive
    Fourier frequencies; I is the periodogram.
    """
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 32 or not np.isfinite(v).all():
        raise ValueError("need >= 32 finite observations")
    if not 0.0 < h < 1.0 or a < 0.0 or sigma <= 0.0:
        raise ValueError("need 0<H<1, a>=0, sigma>0")
    n = v.size
    xd = v - v.mean()
    periodo = np.abs(np.fft.rfft(xd)) ** 2 / n
    freqs = np.fft.rfftfreq(n) * 2 * np.pi
    spec = fou_spectrum(np.abs(freqs), h, a, sigma)
    spec = np.maximum(spec, 1e-300)
    return -0.5 * float(np.sum(np.log(spec) + periodo / spec))


@dataclass(frozen=True)
class FOUEstimate:
    """Joint fOU parameter estimate."""

    h: float
    a: float
    sigma: float
    loglik: float
    converged: bool


def estimate_fou(
    x: FloatArray,
    h_grid: FloatArray | None = None,
    seed: int = 0,
) -> FOUEstimate:
    """Joint Whittle MLE of (H, a, sigma) on a stationary series.

    (H, a) are fit on the Whittle shape likelihood (scale profiled
    out); sigma is then recovered from the Whittle level —
    ``sigma^2 = mean(I / f_1) * C(h, a)`` where ``C = (1/pi) int_0^pi
    f_1`` is the unit-sigma variance over the Nyquist bandwidth. This
    matches the rescale-to-std convention of ``simulate_fou_exact``:
    the sim's spectral level is sigma^2 / C, so the profiled
    periodogram level is sigma^2 / C and multiplying by C recovers
    sigma. Optimization: Nelder-Mead on (H, log a) with seeded
    multistart over the H grid.
    """
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 64 or not np.isfinite(v).all():
        raise ValueError("need >= 64 finite observations")
    n = v.size
    xd = v - v.mean()
    periodo = np.abs(np.fft.rfft(xd)) ** 2 / n
    freqs = np.fft.rfftfreq(n) * 2 * np.pi
    wabs = np.abs(freqs)

    def profiled_nll(h: float, a: float) -> float:
        if not 1e-3 < h < 0.999 or a < 0.0:
            return np.inf
        spec1 = fou_spectrum(wabs, h, a, 1.0)
        spec1 = np.maximum(spec1, 1e-300)
        s2 = float(np.mean(periodo / spec1))  # profiled sigma^2
        if not np.isfinite(s2) or s2 <= 0:
            return np.inf
        return 0.5 * n * float(np.log(s2)) + 0.5 * float(np.sum(np.log(spec1)))

    if h_grid is None:
        h_grid = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    best = (np.inf, 0.5, 0.5)
    rng = np.random.default_rng(seed)
    for h0 in np.asarray(h_grid, dtype=float):
        a0 = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
        res = optimize.minimize(
            lambda th: profiled_nll(th[0], np.exp(th[1])),
            x0=np.array([h0, np.log(max(a0, 1e-3))]),
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-8},
        )
        if res.fun < best[0]:
            best = (float(res.fun), float(res.x[0]), float(np.exp(res.x[1])))
    nll, h_hat, a_hat = best
    spec1 = np.maximum(fou_spectrum(wabs, h_hat, a_hat, 1.0), 1e-300)
    level = float(np.mean(periodo / spec1))
    sigma_hat = float(np.sqrt(max(level * _unit_variance(h_hat, a_hat), 0.0)))
    return FOUEstimate(
        h=h_hat,
        a=a_hat,
        sigma=sigma_hat,
        loglik=-nll,
        converged=bool(np.isfinite(nll)),
    )


def _unit_variance(h: float, a: float) -> float:
    """Variance of the unit-sigma fOU over the Nyquist bandwidth.

    ``C(h, a) = (1/pi) int_0^pi f_1(w) dw`` — trapezoid on a dense
    uniform grid (geometric grids under-resolve the |w|^{1-2H} mass at
    low frequency for H < 1/2). This is the spectral-level-to-variance
    convention ``simulate_fou_exact`` and the periodogram share.
    """
    w = np.linspace(1e-6, np.pi, 8192)
    return float(np.trapezoid(fou_spectrum(w, h, a, 1.0), w) / np.pi)


def fou_autocov_theoretical(lags: FloatArray, h: float, a: float, sigma: float = 1.0) -> FloatArray:
    """fOU autocovariance via spectral inversion (quadrature).

    ``r(tau) = (1/pi) int_0^inf f(w) cos(w tau) dw`` — evaluated by
    trapezoid on a geometric frequency grid, adequate for matching
    statistics in tests/bench.
    """
    tau = np.asarray(lags, dtype=float)
    if not 0.0 < h < 1.0 or a < 0.0 or sigma <= 0.0:
        raise ValueError("need 0<H<1, a>=0, sigma>0")
    w = np.geomspace(1e-4, 1e3, 4000)
    spec = fou_spectrum(w, h, a, sigma)
    out = np.empty(tau.size)
    for i, t in enumerate(tau):
        out[i] = float(np.trapezoid(spec * np.cos(w * t), w)) / np.pi
    return out


def synth_fou_observed(n: int, h: float, a: float, sigma: float = 1.0, seed: int = 0) -> FloatArray:
    """SYNTHETIC stationary fOU observation for recovery tests."""
    return simulate_fou_exact(n, h, a, sigma, seed)


def bench_fractional_ou(seed: int = 20260201) -> dict[str, float]:
    """SYNTHETIC bench for fOU spectral estimation. Correctness only."""
    out: dict[str, float] = {}
    # H recovery across a grid (estimator needs long series; n=2048)
    errs = []
    for h_true in (0.3, 0.5, 0.7):
        x = synth_fou_observed(2048, h_true, a=0.5, sigma=1.0, seed=seed)
        est = estimate_fou(x, seed=seed)
        errs.append(abs(est.h - h_true))
        out[f"synthetic_h_err_{int(h_true * 100)}"] = errs[-1]
    out["synthetic_h_err_mean"] = float(np.mean(errs))
    # sigma recovery at fixed H
    x = synth_fou_observed(2048, 0.4, a=0.5, sigma=1.6, seed=seed + 1)
    est = estimate_fou(x, seed=seed + 1)
    out["synthetic_sigma_relerr"] = float(abs(est.sigma - 1.6) / 1.6)
    out["synthetic_a_est"] = float(est.a)
    # autocovariance match on a shorter realization
    xs = synth_fou_observed(512, 0.35, a=0.8, seed=seed + 2)
    lags = np.arange(0, 8, dtype=float)
    emp = np.array([np.cov(xs[: xs.size - int(t)], xs[int(t) :])[0, 1] for t in lags])
    theo = fou_autocov_theoretical(lags, 0.35, 0.8)
    emp_acf = emp / emp[0]
    theo_acf = theo / theo[0]
    out["synthetic_autocov_l2_relerr"] = float(
        np.linalg.norm(emp_acf - theo_acf) / max(np.linalg.norm(theo_acf), 1e-12)
    )
    # determinism
    e1 = estimate_fou(x, seed=7)
    e2 = estimate_fou(x, seed=7)
    out["synthetic_determinism"] = float(e1.h == e2.h and e1.a == e2.a)
    return out
