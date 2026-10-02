"""Polyphase rational resampling (up P / down Q).

Canonical reference: Crochiere & Rabiner (1983). A windowed-sinc
anti-(image|alias) lowpass at min(1/P, 1/Q) is commuted through the
P-phase decomposition — O(N·len(h)/P) instead of naive interpolate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _design_filter(up: int, down: int, n_taps_per_phase: int = 10) -> FloatArray:
    """Windowed sinc at fc = min(1/up, 1/down)/2 (normalized), len ~ 2·n·max."""
    m = max(up, down)
    n_taps = 2 * n_taps_per_phase * m + 1
    fc = 0.5 / m
    k = np.arange(n_taps) - n_taps // 2
    h = np.sinc(2 * fc * k) * np.blackman(n_taps)
    h = np.asarray(h * (up / np.sum(h)), dtype=np.float64)  # DC unity at up-rate → gain up
    return h


def resample_poly(x: FloatArray, up: int, down: int) -> FloatArray:
    """y ≈ x resampled by up/down."""
    h = _design_filter(up, down)
    half = (len(h) - 1) // 2  # FIR group delay in up-phase units
    # zero-pad to integer number of phases
    pad = (up - len(h) % up) % up
    h = np.concatenate([h, np.zeros(pad)])
    phases = h.reshape(-1, up).T  # phase p → taps h[p::up]
    n_out = int(np.ceil(len(x) * up / down))
    y = np.empty(n_out)
    # y[m] = Σ_k h[m·down − k·up]·x[k]; h indexed by up-phase so
    # phase p = (m·down + half) mod up — `half` removes the FIR delay
    flen = phases.shape[1]
    for m in range(n_out):
        t = m * down + half
        p = t % up
        k0 = t // up
        acc = 0.0
        lo = max(0, k0 - flen + 1)
        for k in range(lo, min(k0 + 1, len(x))):
            acc += phases[p, k0 - k] * x[k]
        y[m] = acc
    return y


def bench_resample_poly(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: 3/2 and 2/3 resample of a bandlimited tone — image
    and alias suppression, endpoint length, spectral fidelity."""
    n = 768
    t = np.arange(n)
    x = np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.11 * t)
    y32 = resample_poly(x, 3, 2)
    y23 = resample_poly(x, 2, 3)
    len32_err = float(abs(len(y32) - 1.5 * n) / n)
    len23_err = float(abs(len(y23) - (2.0 / 3) * n) / n)
    # fidelity: correlate y32 back-downsampled region with x
    t2 = np.arange(len(y32)) * (2.0 / 3)
    ref = np.sin(2 * np.pi * 0.05 * t2) + 0.5 * np.sin(2 * np.pi * 0.11 * t2)
    g = slice(64, -64)
    err32 = float(np.sqrt(np.mean((y32[g] - ref[g]) ** 2)))
    # image suppression: energy above original Nyquist/3 in y32
    spec = np.abs(np.fft.rfft(y32[g]))
    fr = np.fft.rfftfreq(len(y32[g]), 2.0 / 3)
    img = float(spec[fr > 0.4].max() / (spec.max() + 1e-15))
    return {
        "synthetic_resample_len32_err": len32_err,
        "synthetic_resample_len23_err": len23_err,
        "synthetic_resample_err32": err32,
        "synthetic_resample_image": img,
        "synthetic_resample_spec_ok": float(err32 < 0.05 and img < 0.05),
    }
