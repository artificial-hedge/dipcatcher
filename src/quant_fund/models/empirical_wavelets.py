"""Empirical wavelet transform (Gilles 2013).

The EWT builds an *adaptive* wavelet filter bank: segment the Fourier
spectrum of the signal into N bands between successive local minima of
the spectrum magnitude, then apply Littlewood-Paley/Meyer-style
transition filters on each band:

    phi_hat(w)  = 1 on the first band, 0 elsewhere,
    psi_k_hat(w) = smooth indicator on band k

with raised-cosine transitions of width gamma * band-width (gamma
scaled by min band separation). Unlike the fixed dyadic wavelet
decomposition, bands track the signal's actual modes — the adaptive
part is what makes it "empirical".

Honesty: the bench plants two separated tones; the segmentation must
find the valley between them, and the two empirical modes must isolate
the carriers (cross-contamination small). Fail-closed on non-finite
input, a spectrum with too few segments, or empty bands.

References: Gilles (2013) "Empirical wavelet transform", IEEE Trans.
Signal Proc.; Gilles & Heal (2014) "A parameterless scale-space
approach to find meaningful modes in histograms".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _spectrum_minima(mag: FloatArray, n_bands: int) -> FloatArray:
    """Segment boundaries: the n_bands-1 deepest local minima between
    detected peaks, plus the endpoints (bin indices)."""
    n = mag.size
    # smooth lightly so trivial ripples don't count as minima
    k = max(3, n // 128)
    sm = np.convolve(mag, np.ones(k) / k, mode="same")
    peaks = np.where((sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:]))[0] + 1
    if peaks.size < 2:
        return np.array([0, n], dtype=np.float64)
    # keep only the prominent peaks so noise ripples don't seed
    # spurious valleys
    prom = sm[peaks] >= 0.15 * sm.max()
    peaks = peaks[prom]
    if peaks.size < 2:
        return np.array([0, n], dtype=np.float64)
    # candidate valleys between consecutive prominent peaks,
    # scored by depth relative to the shallower flanking peak
    cand = []
    for i in range(peaks.size - 1):
        lo, hi = peaks[i], peaks[i + 1]
        v = int(lo + np.argmin(sm[lo:hi]))
        depth = min(sm[lo], sm[hi]) - sm[v]
        cand.append((depth, v))
    cand.sort(reverse=True)
    keep = sorted(v for _, v in cand[: n_bands - 1])
    return np.asarray(np.unique([0] + keep + [n]), dtype=np.float64)


def _raised_cosine(t: FloatArray) -> FloatArray:
    """Meyer transition function on [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.sin(np.pi * (t - 0.5))), dtype=np.float64)


def ewt(
    x: FloatArray,
    n_bands: int = 3,
    gamma: float = 0.5,
) -> dict[str, FloatArray]:
    """Empirical wavelet transform.

    Returns per-band time-domain modes, the band boundaries (in
    cycles/sample), and the filter magnitude rows used.
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 64 or not np.isfinite(v).all():
        raise ValueError("series too short or non-finite")
    if not 2 <= n_bands <= 8:
        raise ValueError("bad n_bands")
    xh = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(n)
    mag = np.abs(xh)
    edges = _spectrum_minima(mag, n_bands).astype(int)
    if edges.size < 3:
        raise ValueError("spectrum not segmentable")
    edges = edges[: n_bands + 1] if edges.size > n_bands + 1 else edges
    n_real = edges.size - 1
    filters = np.zeros((n_real, freqs.size))
    for b in range(n_real):
        lo, hi = edges[b], edges[b + 1]
        # Gilles-style transitions CENTERED on each boundary: the
        # filter rises across [lo - tau, lo + tau] and falls across
        # [hi - tau, hi + tau], leaving a flat passband inside. tau
        # is a small fixed fraction so a valley sitting a few bins
        # from a peak doesn't eat the carrier.
        tau_lo = max(2, min(8, int(gamma * (hi - lo) // 4), lo))
        tau_hi = max(2, min(8, int(gamma * (hi - lo) // 4), freqs.size - hi))
        f = np.ones(freqs.size)
        in_r = _raised_cosine((np.arange(freqs.size) - (lo - tau_lo)) / (2 * tau_lo))
        out_r = _raised_cosine(((hi + tau_hi) - np.arange(freqs.size)) / (2 * tau_hi))
        f = np.minimum(in_r, out_r)
        f[: max(0, lo - tau_lo)] = 0.0
        f[hi + tau_hi :] = 0.0
        filters[b] = f
    modes = np.fft.irfft(filters * xh[None, :], n=n, axis=1)
    return {
        "modes": np.asarray(modes, dtype=np.float64),
        "edges_cycles": np.asarray(freqs[edges[:n_real]], dtype=np.float64),
        "filters": np.asarray(filters, dtype=np.float64),
    }


def bench_empirical_wavelets(seed: int = 20261231 + 409) -> dict[str, float]:
    """SYNTHETIC check — two tones separated into distinct modes."""
    rng = np.random.default_rng(seed)
    n = 1024
    t = np.arange(n)
    f1, f2 = 0.08, 0.25
    x = (
        np.cos(2 * np.pi * f1 * t)
        + 0.8 * np.cos(2 * np.pi * f2 * t + 0.4)
        + 0.05 * rng.standard_normal(n)
    )
    out = ewt(x, n_bands=3)
    modes = np.asarray(out["modes"])
    if modes.shape[0] < 2:
        raise ValueError("too few modes returned")
    # dominant freq of each mode vs planted tones
    mags = np.abs(np.fft.rfft(modes, axis=1))
    freqs = np.fft.rfftfreq(n)
    dom = freqs[np.argmax(mags, axis=1)]
    planted = np.sort(np.array([f1, f2]))
    # match each planted tone to some mode within tolerance
    err = max(min(abs(d - p) for d in dom) for p in planted)
    if err > 0.03:
        raise ValueError(f"EWT mode coverage off: {dom}")
    # isolation: energy of the f1 mode inside the f2 band should be small
    band2 = (freqs > 0.2) & (freqs < 0.3)
    i_low = int(np.argmin(np.abs(dom - f1)))
    leak = float(np.sum(mags[i_low][band2] ** 2) / np.sum(mags[i_low] ** 2))
    if leak > 0.05:
        raise ValueError(f"cross-mode leakage: {leak}")
    return {
        "synthetic_ewt_cover_err": float(err),
        "synthetic_ewt_leakage": leak,
        "synthetic_ewt_n_bands": float(modes.shape[0]),
        "synthetic_score": 1.0,
    }
