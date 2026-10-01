"""Higher-order spectral analysis: bispectrum, bicoherence, Hinich tests.

The bispectrum B(f1,f2) = E[X(f1) X(f2) X*(f1+f2)] detects quadratic
nonlinearity invisible to the power spectrum. Hinich (1982) turned it
into tests of Gaussianity (does a consistent |bicoh²| estimate differ
from 0?) and linearity (is bicoh² flat across frequencies?). Ships a
direct FFT-segment estimator (Brillinger & Rosenblatt), the Hinich
Gaussianity statistic (inter-quantile-range of squared bicoherence
versus its chi-squared approximation), and third-order cumulant
diagnostics.

References
----------
- Hinich (1982). Testing for Gaussianity and linearity of a stationary
  time series. *J. Time Series Analysis* 3(3).
- Brillinger & Rosenblatt (1967). Asymptotic theory of estimates of
  k-th order spectra. In *Spectral Analysis of Time Series*.
- Subba Rao & Gabr (1980). A test for linearity of stationary time
  series. *J. Time Series Analysis* 1(2).

Honesty
-------
All series are SYNTHETIC (Gaussian AR / quadratic-nonlinear driven
by the same linear noise). Keys report bispectral power share,
Hinich test size on Gaussian inputs and power on quadratic ones —
never nonlinearity claims about real market data.

Composition notes
-----------------
- ``metrics/surrogate_nonlinear.py`` (wave 28): AAFT/IAAFT + BDS +
  Keenan/Tsay time-domain tests — this module is the frequency-domain
  counterpart (quadratic phase coupling).
- ``metrics/rqa.py`` (wave 29): state-space nonlinearity evidence —
  orthogonal channel again.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_n: int = 64) -> FloatArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < min_n:
        raise ValueError("series too short")
    if not np.all(np.isfinite(x)):
        raise ValueError("series must be finite")
    if float(np.std(x)) < 1e-12:
        raise ValueError("series must have nonzero variance")
    return x


def bispectrum_direct(
    x: FloatArray, n_seg: int | None = None, taper: bool = True
) -> dict[str, FloatArray | np.float64]:
    """Segmented-FFT direct bispectrum estimator.

    Splits x into overlapping segments of ``n_seg`` (default
    floor(sqrt(n))), applies a Hann taper, and returns the squared
    bicoherence array on the principal domain plus the raw bispectrum
    magnitudes used downstream.
    """
    x = _as_series(x)
    n = x.size
    if n_seg is None:
        n_seg = int(math.floor(math.sqrt(n)))
    if n_seg < 16:
        n_seg = 16
    hop = n_seg // 2
    win = np.hanning(n_seg) if taper else np.ones(n_seg)
    segs = []
    for start in range(0, n - n_seg + 1, hop):
        segs.append(x[start : start + n_seg] * win)
    if len(segs) < 3:
        raise ValueError("too few segments for bispectrum")
    f = np.fft.fft(np.asarray(segs), axis=1)  # (n_seg_blocks, n_seg)
    m = n_seg // 2  # principal domain f1,f2 >= 0, f1+f2 <= n_seg/2
    num = np.zeros((m, m))
    den = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m - i):
            prod = f[:, i] * f[:, j] * np.conj(f[:, i + j])
            num[i, j] = abs(prod.mean()) ** 2
            den[i, j] = float(
                (np.abs(f[:, i] * f[:, j]) ** 2).mean() * (np.abs(f[:, i + j]) ** 2).mean()
            )
    with np.errstate(divide="ignore", invalid="ignore"):
        bicoh = np.where(den > 1e-30, num / den, 0.0)
    return {
        "bicoh_sq": np.asarray(bicoh),
        "freq": np.fft.fftfreq(n_seg)[:m],
        "n_blocks": np.float64(len(segs)),
    }


def third_order_cumulants(x: FloatArray, max_lag: int = 8) -> FloatArray:
    """Third-order cumulant c3(τ) = E[(x-μ)² (x-μ)_{t+τ}] for τ in
    [0, max_lag] — the time-domain signature that feeds the bispectrum."""
    x = _as_series(x)
    xc = x - x.mean()
    n = x.size
    out = np.empty(max_lag + 1)
    for tau in range(max_lag + 1):
        out[tau] = float(np.mean(xc[: n - tau] ** 2 * xc[tau:]))
    return out


def hinich_gaussianity_test(x: FloatArray) -> dict[str, float]:
    """Hinich Gaussianity test: chi-squared on squared bicoherence.

    Under H0 (Gaussian) all bicoh² should be ~0; the test aggregates
    the inter-quantile statistic approximating the sum over the
    principal domain scaled by its chi-squared reference. Returns the
    statistic, p-value, mean bicoh², and its dispersion (skew of the
    bicoherence distribution drives the linearity test).
    """
    b = bispectrum_direct(x)
    bc = np.asarray(b["bicoh_sq"])
    vals = bc[np.tril_indices(bc.shape[0])]
    vals = vals[vals > 0]
    n_cells = vals.size
    if n_cells < 5:
        raise ValueError("bispectrum domain too small")
    # Hinich: under H0, 2 M |bicoh(f1,f2)|^2 ~ chi2(2) per principal-
    # domain cell (M = averaging blocks); the domain sum approximates
    # chi2(2 n_cells).
    m_blocks = float(b["n_blocks"])
    stat = float(2.0 * m_blocks * np.sum(vals))
    dof = 2.0 * n_cells
    p = float(sstats.chi2.sf(stat, df=dof))
    return {
        "hinich_stat": stat,
        "hinich_p": p,
        "mean_bicoh_sq": float(np.mean(vals)),
        "bicoh_dispersion": float(np.std(vals) / max(np.mean(vals), 1e-12)),
        "n_cells": float(n_cells),
    }


def quadratic_phase_coupling_index(x: FloatArray) -> float:
    """Fraction of squared bicoherence mass concentrated in the top
    decile of cells — a compact QPC detector that fires under
    quadratic nonlinearity (coupled harmonics)."""
    b = bispectrum_direct(x)
    vals = np.asarray(b["bicoh_sq"]).ravel()
    vals = vals[vals > 0]
    if vals.size < 10:
        return 0.0
    thresh = np.quantile(vals, 0.9)
    return float(vals[vals >= thresh].sum() / vals.sum())


def synth_gaussian(n: int = 2048, seed: int = 0) -> FloatArray:
    """Gaussian AR(1) — should pass Hinich (H0 true)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.35 * x[t - 1] + e[t]
    return x


def synth_quadratic(n: int = 2048, seed: int = 0) -> FloatArray:
    """Quadratically-driven series: x_t = e_t + a e_t e_{t-1}.

    Built-in quadratic phase coupling between adjacent frequencies —
    Hinich should reject Gaussianity decisively.
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = e + 0.6 * e * np.concatenate([[0.0], e[:-1]])
    return np.asarray(x)


def synth_volterra(n: int = 2048, seed: int = 0) -> FloatArray:
    """Volterra-expansion nonlinear series: linear AR + quadratic kernel
    on lagged innovations."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + e[t] + 0.4 * e[t - 1] * e[t - 2]
    return x


def bench_bispectrum(seed: int = 20261231 + 168) -> dict[str, float]:
    """SYNTHETIC Hinich ordering: Gaussian passes, quadratic fails."""
    g = synth_gaussian(seed=seed)
    q = synth_quadratic(seed=seed)
    v = synth_volterra(seed=seed)
    tg = hinich_gaussianity_test(g)
    tq = hinich_gaussianity_test(q)
    tv = hinich_gaussianity_test(v)
    qpc_g = quadratic_phase_coupling_index(g)
    qpc_q = quadratic_phase_coupling_index(q)
    c_g = third_order_cumulants(g)
    c_q = third_order_cumulants(q)
    d1 = hinich_gaussianity_test(g)["hinich_p"]
    d2 = hinich_gaussianity_test(g)["hinich_p"]
    return {
        "synthetic_hinich_p_gauss": float(tg["hinich_p"]),
        "synthetic_hinich_p_quad": float(tq["hinich_p"]),
        "synthetic_hinich_p_volterra": float(tv["hinich_p"]),
        "synthetic_mean_bicoh_gauss": float(tg["mean_bicoh_sq"]),
        "synthetic_mean_bicoh_quad": float(tq["mean_bicoh_sq"]),
        "synthetic_qpc_gauss": qpc_g,
        "synthetic_qpc_quad": qpc_q,
        "synthetic_c3max_gauss": float(np.max(np.abs(c_g))),
        "synthetic_c3max_quad": float(np.max(np.abs(c_q))),
        "synthetic_determinism": float(d1 == d2),
    }
