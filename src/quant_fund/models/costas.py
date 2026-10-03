"""Costas canon: Costas-loop carrier recovery for BPSK/QPSK —
phase-detector (sign × sample) feeding a second-order loop
filter on a numerically controlled oscillator. Bench: static
phase offset locked, small frequency offset tracked, and
decision BER vs unrecovered baseline. All SYNTHETIC.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def bpsk_symbols(n: int, seed: int = 0) -> FloatArray:
    rng = np.random.default_rng(seed)
    return (2 * rng.integers(0, 2, n) - 1).astype(np.float64)


def costas_loop(
    rx: FloatArray,
    fs: float = 8.0,
    loop_bw: float = 0.02,
    order: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """BPSK Costas loop on complex baseband samples at fs
    samples/symbol.

    Returns (decisions, phase_trace, freq_trace).
    """
    rx = np.asarray(rx, dtype=np.complex128)
    # loop filter constants (critical damping)
    damp = 0.707
    t1 = loop_bw / (damp + 1 / (4 * damp))
    k1 = 4 * t1 / (1 + 2 * damp * t1 + t1 * t1)
    k2 = 4 * t1 * t1 / (1 + 2 * damp * t1 + t1 * t1)
    phase = 0.0
    freq = 0.0
    out = np.zeros(len(rx), dtype=np.float64)
    phases = np.zeros(len(rx))
    freqs = np.zeros(len(rx))
    for i, s in enumerate(rx):
        # rotate by -phase
        x = s * np.exp(-1j * phase)
        d = 1.0 if x.real >= 0 else -1.0
        out[i] = d
        # BPSK phase detector: e = Re(x)·Im(x)
        e = float(x.real * x.imag)
        freq += k2 * e
        phase += freq + k1 * e
        phases[i] = phase
        freqs[i] = freq
    return out, phases, freqs


def costas_qpsk(rx: FloatArray, loop_bw: float = 0.02) -> tuple[NDArray[np.complex128], FloatArray]:
    """QPSK Costas variant: decision-directed phase detector."""
    rx = np.asarray(rx, dtype=np.complex128)
    damp = 0.707
    t1 = loop_bw / (damp + 1 / (4 * damp))
    k1 = 4 * t1 / (1 + 2 * damp * t1 + t1 * t1)
    k2 = 4 * t1 * t1 / (1 + 2 * damp * t1 + t1 * t1)
    phase = 0.0
    freq = 0.0
    out = np.zeros(len(rx), dtype=np.complex128)
    ph = np.zeros(len(rx))
    for i, s in enumerate(rx):
        x = s * np.exp(-1j * phase)
        di = np.sign(x.real)
        dq = np.sign(x.imag)
        e = float(di * x.imag - dq * x.real)
        out[i] = complex(di, dq)
        freq += k2 * e
        phase += freq + k1 * e
        ph[i] = phase
    return out, ph


def bench_costas(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    n = 600
    bits = bpsk_symbols(n, seed=seed + 1)
    # static phase offset
    off = 0.7
    rx = bits * np.exp(1j * off)
    rx += rng.normal(0, 0.15, n) + 1j * rng.normal(0, 0.15, n)
    dec, phases, freqs = costas_loop(rx)
    # skip transient (first 20%)
    st = n // 5
    out["synthetic_costas_static_ber"] = float(np.mean(dec[st:] != bits[st:]))
    # unrecovered baseline (direct sign decision)
    base = (np.real(rx) > 0).astype(np.float64) * 2 - 1
    out["synthetic_costas_baseline_ber"] = float(np.mean(base[st:] != bits[st:]))
    # frequency offset: δf = 0.002 rad/sample
    delf = 0.002
    rx2 = bits * np.exp(1j * (off + delf * np.arange(n)))
    rx2 += rng.normal(0, 0.15, n) + 1j * rng.normal(0, 0.15, n)
    dec2, ph2, fr2 = costas_loop(rx2)
    out["synthetic_costas_freq_ber"] = float(np.mean(dec2[st:] != bits[st:]))
    out["synthetic_costas_freq_est_err"] = float(abs(fr2[-1] - delf))
    # residual phase jitter after lock
    resid = phases[st:] - off
    resid = (resid + math.pi) % (2 * math.pi) - math.pi
    out["synthetic_costas_resid_rms"] = float(np.sqrt(np.mean(resid**2)))
    return out
