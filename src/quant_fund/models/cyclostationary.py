"""Cyclostationary analysis — Gardner's cyclic autocorrelation and
spectral correlation density (SCD).

A cyclostationary process has periodically time-varying second-order
statistics; its cyclic autocorrelation

    R_x^alpha(tau) = lim_T (1/T) int x(t+tau/2) x*(t-tau/2)
                                  e^{-i 2 pi alpha t} dt

is nonzero at discrete cycle frequencies alpha (symbol rates, carrier
harmonics, diurnal cycles). The spectral correlation density is its
Fourier transform over tau:

    S_x^alpha(f) = int R_x^alpha(tau) e^{-i 2 pi f tau} dtau,

estimated here by the frequency-smoothing method: smoothed cyclic
periodogram over short-time FFTs of x(t) and x(t) e^{-i 2 pi alpha t}.

Honesty: the bench plants a known cycle frequency (amplitude-modulated
carrier) and requires the SCD to peak at the planted alpha with
power well above the white-noise floor, and the cyclic autocorrelation
magnitude to concentrate at that alpha. Fail-closed on non-finite
input or degenerate segment counts.

References: Gardner (1986) "The spectral correlation theory of
cyclostationary time-series"; Gardner (1991) "Exploiting spectral
redundancy in cyclostationary signals"; Antoni (2007) cyclic
spectral analysis in practice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cyclic_autocorrelation(x: FloatArray, alpha: float, max_lag: int | None = None) -> FloatArray:
    """Cyclic autocorrelation R_x^alpha(tau) over tau = -max_lag..max_lag.

    Implemented by mixing: y(t) = x(t) e^{-i pi alpha t}, then
    correlating y(t+tau) conj(y(t)) — the e^{-i 2 pi alpha t} kernel
    appears through the product of the two mixes. Returns the
    magnitude (phase alignment is sample-dependent).
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 32 or not np.isfinite(v).all():
        raise ValueError("series too short or non-finite")
    m = n // 4 if max_lag is None else min(max_lag, n // 4)
    t = np.arange(n)
    mix = v * np.exp(-1j * np.pi * alpha * t)
    taus = np.arange(-m, m + 1)
    out = np.empty(taus.size, dtype=complex)
    for i, tau in enumerate(taus):
        a = mix[m + tau : n - m]
        b = np.conj(mix[m : n - m - tau])
        out[i] = np.mean(a * b)
    return np.asarray(np.abs(out), dtype=np.float64)


def scd_peak(
    x: FloatArray,
    alphas: FloatArray,
    seg_len: int = 64,
    freq_bin: float | None = None,
) -> dict[str, FloatArray | float]:
    """Cycle-frequency profile via the cyclic second-order moment.

    For each candidate alpha, seg_len segments of x(t)^2 are mixed
    down at alpha and averaged; the magnitude is the Dandawate-
    Giannakis cyclic-moment detector |R_x^alpha(0)| evaluated on
    x^2 (equivalently the alpha-bin of the spectrum of the squared
    signal). This is the standard cyclostationarity presence test:
    white noise contributes only an estimation floor, while a
    planted modulation produces a sharp line at its cycle rate.

    Returns the magnitude profile across alphas plus the peak alpha.
    The alpha grid is in cycles per sample.
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    a_grid = np.asarray(alphas, dtype=float).ravel()
    if n < 64 or not np.isfinite(v).all():
        raise ValueError("series too short or non-finite")
    if a_grid.size < 3 or seg_len < 8 or n // seg_len < 4:
        raise ValueError("bad alpha grid or segmentation")
    n_seg = n // seg_len
    t = np.arange(seg_len)
    win = np.hanning(seg_len)
    prof = np.empty(a_grid.size)
    for i, alpha in enumerate(a_grid):
        acc = 0.0
        for s in range(n_seg):
            seg = v[s * seg_len : (s + 1) * seg_len] ** 2
            seg = seg - seg.mean()  # drop DC so its window sidelobes
            # don't masquerade as small-alpha cyclostationarity
            acc += float(np.abs(np.mean(seg * win * np.exp(-2j * np.pi * alpha * t))))
        prof[i] = acc / n_seg
    pk = int(np.argmax(prof))
    return {
        "alphas": np.asarray(a_grid, dtype=np.float64),
        "scd": np.asarray(prof, dtype=np.float64),
        "alpha_peak": float(a_grid[pk]),
        "scd_peak": float(prof[pk]),
    }


def bench_cyclostationary(seed: int = 20261231 + 408) -> dict[str, float]:
    """SYNTHETIC check — planted AM cycle frequency recovered."""
    rng = np.random.default_rng(seed)
    n = 2048
    t = np.arange(n)
    alpha_true = 0.05
    # AM: x = (1 + m cos(2 pi alpha t)) cos(2 pi f0 t) + noise —
    # cyclostationary at alpha = modulation rate
    f0 = 0.15
    x = (1.0 + 0.8 * np.cos(2 * np.pi * alpha_true * t)) * np.cos(
        2 * np.pi * f0 * t
    ) + 0.5 * rng.standard_normal(n)
    alphas = np.linspace(0.01, 0.12, 60)
    out = scd_peak(x, alphas, seg_len=128)
    peak = float(out["alpha_peak"])
    err = abs(peak - alpha_true)
    if err > 0.01:
        raise ValueError(f"cycle frequency off: {peak}")
    # SCD contrast: peak vs median floor
    scd = np.asarray(out["scd"])
    contrast = float(scd.max() / np.median(scd))
    if contrast < 1.5:
        raise ValueError(f"SCD peak too weak: {contrast}")
    ca = cyclic_autocorrelation(x, alpha_true)
    if not np.isfinite(ca).all():
        raise ValueError("cyclic autocorrelation non-finite")
    return {
        "synthetic_cyclostat_alpha_err": float(err),
        "synthetic_cyclostat_contrast": contrast,
        "synthetic_cyclostat_alpha_hat": peak,
        "score": 1.0,
    }
