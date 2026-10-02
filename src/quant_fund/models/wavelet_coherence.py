"""Torrence-Compo cross-wavelet coherence (Morlet CWT).

References
----------
- Torrence, C. & Compo, G.P. (1998). "A Practical Guide to
  Wavelet Analysis." *Bulletin of the American Meteorological
  Society* 79(1), 61-78.
- Grinsted, A., Moore, J.C. & Jevrejeva, S. (2004).
  "Application of the Cross Wavelet Transform and Wavelet
  Coherence to Geophysical Time Series." *Nonlinear Processes
  in Geophysics* 11, 561-566.
- Maraun, D. & Kurths, J. (2004). "Cross Wavelet Analysis:
  Significance Testing and Pitfalls." *Nonlinear Processes in
  Geophysics* 11, 505-514.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Wavelet coherence localizes correlation in time-frequency:
``R^2(s, tau) = |S(s^{-1} W_xy)|^2 / (S(s^{-1} |W_x|^2) *
S(s^{-1} |W_y|^2))`` where ``S`` smooths over both scale and
time — coherence without smoothing is trivially 1 everywhere,
which is the classic pitfall (Maraun-Kurths). We use a Morlet
CWT via FFT convolution (``omega0 = 6`` analytic Morlet) at
log-spaced scales, boxcar scale-smoothing plus a scale-
dependent time-smoothing window of width ~scale (the
Torrence-Compo recipe), and report both the coherence matrix
and the band-mean coherence. ``synth_coherence`` plants a
shared narrowband component in a mid-frequency band plus
independent noise; the bench gates on coherence concentrating
in the planted band and vanishing in the unplanted bands.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _as_series(x: FloatArray, min_len: int = 128) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _morlet_cwt(
    x: FloatArray,
    scales: FloatArray,
    omega0: float = 6.0,
) -> ComplexArray:
    """FFT Morlet CWT at given scales (periods in samples)."""
    n = x.size
    xw = np.fft.fft(x)
    freqs = 2 * np.pi * np.fft.fftfreq(n)
    out = np.empty((scales.size, n), dtype=np.complex128)
    for i, s in enumerate(scales):
        # daughter wavelet in freq domain (Torrence-Compo eq)
        daughter = (np.pi**-0.25) * np.sqrt(n / s) * np.exp(-((s * freqs - omega0) ** 2) / 2.0)
        out[i] = np.fft.ifft(xw * daughter)
    return out


def wavelet_coherence(
    x: FloatArray,
    y: FloatArray,
    n_scales: int = 24,
    smin: float = 4.0,
    smax: float | None = None,
) -> dict[str, FloatArray]:
    """Torrence-Compo wavelet coherence matrix."""
    vx = _as_series(x)
    vy = _as_series(y)
    if vx.size != vy.size:
        raise ValueError("length mismatch")
    n = vx.size
    smax = smax or n / 4.0
    if smax <= smin or n_scales < 4:
        raise ValueError("bad scales")
    scales = np.geomspace(smin, smax, n_scales)
    wx = _morlet_cwt(vx, scales)
    wy = _morlet_cwt(vy, scales)
    sxx = np.abs(wx) ** 2
    syy = np.abs(wy) ** 2
    sxy = wx * np.conj(wy)
    # smooth: boxcar over scales + scale-dependent time window
    sm_sxx = np.zeros_like(sxx)
    sm_syy = np.zeros_like(syy)
    sm_sxy = np.zeros_like(sxy)
    for i, s in enumerate(scales):
        tw = max(3, int(round(s / 2)))
        ker = np.ones(tw) / tw
        sm_sxx[i] = np.convolve(sxx[i], ker, "same")
        sm_syy[i] = np.convolve(syy[i], ker, "same")
        sm_sxy[i] = np.convolve(sxy[i], ker, "same")

    # scale smoothing (moving average over neighboring scales)
    def _scsm(a: ComplexArray | FloatArray) -> ComplexArray | FloatArray:
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            lo = max(0, i - 1)
            hi = min(a.shape[0], i + 2)
            out[i] = a[lo:hi].mean(axis=0)
        return out

    sm_sxx = np.asarray(_scsm(sm_sxx))
    sm_syy = np.asarray(_scsm(sm_syy))
    sm_sxy = np.asarray(_scsm(sm_sxy))
    coh = np.abs(sm_sxy) ** 2 / np.maximum(sm_sxx * sm_syy, 1e-30)
    coh = np.clip(coh, 0.0, 1.0)
    out: dict[str, FloatArray] = {
        "scales": scales,
        "coherence": np.asarray(coh, dtype=np.float64),
        "mean_coherence": np.array([float(np.mean(coh))]),
        "max_coherence": np.array([float(np.max(coh))]),
    }
    return out


def synth_coherence(
    seed: int = 20261231 + 345,
    n: int = 1024,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC shared mid-band component + independent noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shared = np.sin(2 * np.pi * t / 32.0)
    x = shared + 0.6 * rng.standard_normal(n)
    y = shared + 0.6 * rng.standard_normal(n)
    indep = rng.standard_normal(n)
    return x.astype(np.float64), y.astype(np.float64), indep.astype(np.float64)


def bench_wavelet_coherence(seed: int = 20261231 + 345) -> dict[str, float]:
    x, y, indep = synth_coherence(seed=seed)
    r = wavelet_coherence(x, y, n_scales=24)
    r_n = wavelet_coherence(x, indep, n_scales=24)
    scales = np.asarray(r["scales"])
    band = (scales >= 24.0) & (scales <= 48.0)
    planted = float(np.mean(np.asarray(r["coherence"])[band]))
    out_band = float(np.mean(np.asarray(r["coherence"])[~band]))
    null_band = float(np.mean(np.asarray(r_n["coherence"])[band]))
    ok = planted > 0.7 and planted > null_band + 0.2 and planted > out_band + 0.2
    out: dict[str, float] = {
        "synthetic_wcoh_planted_band": planted,
        "synthetic_wcoh_off_band": out_band,
        "synthetic_wcoh_null_band": null_band,
        "synthetic_wcoh_mean": float(r["mean_coherence"][0]),
        "score": 1.0 if ok else 0.0,
    }
    return out
