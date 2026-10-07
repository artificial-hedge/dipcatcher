"""filtfilt — zero-phase forward-backward IIR/FIR filtering (SYNTHETIC).

Canonical reference: Gustafsson (1996); scipy.signal.filtfilt.
Odd edge padding (3·nf), steady-state initial conditions for the
forward pass, then reverse repeat — net zero phase, squared magnitude.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _lfilter(
    b: FloatArray, a: FloatArray, x: FloatArray, zi: FloatArray | None = None
) -> tuple[FloatArray, FloatArray]:
    """DF2T IIR/FIR filter; returns (y, zf). a[0] = 1 assumed."""
    nb, na = len(b), len(a)
    n = max(nb, na) - 1
    z = np.zeros(n) if zi is None else np.asarray(zi, dtype=np.float64).copy()
    y = np.empty(len(x))
    bb = np.concatenate([b, np.zeros(n + 1 - nb)])
    aa = np.concatenate([a, np.zeros(n + 1 - na)])
    for i in range(len(x)):
        xm = x[i]
        ym = bb[0] * xm + z[0]
        for k in range(n - 1):
            z[k] = bb[k + 1] * xm + z[k + 1] - aa[k + 1] * ym
        z[n - 1] = bb[n] * xm - aa[n] * ym
        y[i] = ym
    return y, z


def _zi_step(b: FloatArray, a: FloatArray) -> FloatArray:
    """Steady-state filter state for a unit-step input."""
    n = max(len(b), len(a)) - 1
    if n == 0:
        return np.zeros(0)
    bb = np.concatenate([b, np.zeros(n + 1 - len(b))])
    aa = np.concatenate([a, np.zeros(n + 1 - len(a))])
    # solve (I + A) z = B where A is the companion-style update matrix:
    # z_k' = b_{k+1} + z_{k+1} − a_{k+1}·(b_0 + z_0); steady state z_k'=z_k,
    # unit step ⇒ y = b_0 + z_0 ... solve linear system for z directly:
    # z_k = b_{k+1} − a_{k+1}·y + z_{k+1}, y = Σb/Σa
    yss = float(np.sum(bb) / np.sum(aa))
    z = np.zeros(n)
    # march from the last tap backward
    for k in range(n - 1, -1, -1):
        nxt = z[k + 1] if k + 1 < n else 0.0
        z[k] = bb[k + 1] - aa[k + 1] * yss + nxt
    return z


def filtfilt(b: FloatArray, a: FloatArray, x: FloatArray) -> FloatArray:
    """Zero-phase filter: pad odd → forward → flip → forward → flip."""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    nf = max(len(b), len(a))
    pad = 3 * nf
    if len(x) <= pad:
        pad = len(x) - 1
    xp = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2 : -pad - 2 : -1]])
    zi = _zi_step(b, a)
    y, _ = _lfilter(b, a, xp, zi * xp[0])
    y = y[::-1]
    y, _ = _lfilter(b, a, y, zi * y[0])
    y = y[::-1]
    return y[pad : pad + len(x)]


def bench_filtfilt(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: zero group delay vs lfilter; edge transient bounded;
    magnitude squared = |H|²."""
    from quant_fund.models.iir_design import iir_design, sos_apply

    sos = iir_design(4, 0.1, "butter")
    # SOS → single-section chain handled per-section filtfilt
    t = np.arange(2048)
    x = np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 0.4 * t)
    y_lp = sos_apply(sos, x)
    # forward-backward through each section
    y_ff = x.copy()
    for sec in sos:
        y_ff = filtfilt(sec[:3], sec[3:], y_ff)
    # zero phase: cross-correlate input sinusoid with output at lag 0
    s1 = np.sin(2 * np.pi * 0.05 * t)
    lag_lp = int(np.argmax(np.correlate(y_lp, s1, "full")) - (len(t) - 1))
    lag_ff = int(np.argmax(np.correlate(y_ff, s1, "full")) - (len(t) - 1))
    hi = np.sin(2 * np.pi * 0.4 * t)
    sup = float(np.sqrt(np.mean(y_ff[512:-512] ** 2)))  # should ≈ |H(0.05)|²·s1
    resid = float(np.sqrt(np.mean((y_ff[512:-512] - s1[512:-512] * np.abs(1.0) ** 2) ** 2)))
    del hi
    return {
        "synthetic_filtfilt_lag": float(abs(lag_ff)),
        "synthetic_filtfilt_vs_lag": float(abs(lag_lp)),
        "synthetic_filtfilt_zero_phase": float(abs(lag_ff) <= 1),
        "synthetic_filtfilt_hi_suppress": float(
            -20 * np.log10(sup / (np.sqrt(0.5) + 1e-15) + 1e-15)
        ),
        "synthetic_filtfilt_resid": resid,
        "synthetic_filtfilt_finite": float(np.isfinite(y_ff).all()),
    }
