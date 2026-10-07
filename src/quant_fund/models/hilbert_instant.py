"""Hilbert-transform instantaneous amplitude/frequency of the chirp (SYNTHETIC).

Analytic signal z = x + i H(x) via the Hilbert FIR/FFT; instantaneous
frequency = d(phase)/dt. Bench: |z| envelope smoothness and median
instantaneous-frequency error vs the planted chirp law (edges trimmed).
"""

import numpy as np

from quant_fund.models._sig3_synth import FS, chirp


def _analytic(x: np.ndarray) -> np.ndarray:
    n = len(x)
    xf = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : n // 2] = 2.0
    h[n // 2] = 1.0
    return np.fft.ifft(xf * h)


def bench_hilbert_instant(seed: int = 4807) -> dict[str, float]:
    x = chirp(seed)
    z = _analytic(x)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst = np.diff(phase) * FS / (2 * np.pi)
    t = np.arange(len(x)) / FS
    true_f = 200.0 + 600.0 * t / (len(x) / FS)
    edge = len(x) // 10
    m = min(len(inst), len(true_f) - 1)
    sl = slice(edge, m - edge)
    rel = np.abs(inst[sl] - true_f[1:][sl]) / true_f[1:][sl]
    env_cv = float(np.std(env[edge:-edge]) / np.mean(env[edge:-edge]))
    return {
        "synthetic_hil_med_err": float(np.median(rel)),
        "synthetic_hil_env_cv": env_cv,
        "synthetic_hil_env_mean": float(np.mean(env[edge:-edge])),
        "synthetic_hil_inst_start": float(inst[edge]),
        "synthetic_hil_inst_end": float(inst[-edge]),
    }
