"""Thin-bed spectral decomposition / tuning-thickness estimation.

An equal-and-opposite reflection doublet (+r at t0, -r at t0+tau) has
amplitude spectrum 2|sin(pi f tau)| with notches at f = n/tau; the first
spectral notch recovers bed thickness.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 934


def thin_bed_trace(
    refl_amp: float,
    tau_samples: int,
    n: int,
    wavelet: np.ndarray,
    t0: int = 60,
) -> np.ndarray:
    spikes = np.zeros(n)
    spikes[t0] = refl_amp
    spikes[t0 + tau_samples] = -refl_amp
    return np.convolve(spikes, wavelet)[:n]


def first_notch_freq(trace: np.ndarray, dt: float) -> float:
    spec = np.abs(np.fft.rfft(trace))
    freqs = np.fft.rfftfreq(len(trace), dt)
    j = int(np.argmax(spec))
    trough = j
    while trough < len(spec) - 1 and spec[trough + 1] <= spec[trough]:
        trough += 1
    if trough >= len(spec) - 2:
        return float("nan")
    for k in range(trough, len(spec) - 2):
        if spec[k] <= spec[k - 1] and spec[k] <= spec[k + 1] and spec[k] < 0.15 * spec[j]:
            return float(freqs[k])
    return float("nan")


def tuning_thickness(trace: np.ndarray, dt: float) -> float:
    f_n = first_notch_freq(trace, dt)
    return 1.0 / f_n


def _ricker(f: float, dt: float) -> np.ndarray:
    h = int(np.ceil(1.5 / (f * dt)))
    t = np.arange(-h, h + 1) * dt
    a = np.pi**2 * f**2 * t**2
    return (1.0 - 2.0 * a) * np.exp(-a)


def bench_spectral_decomp(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dt = 0.001
    w = _ricker(30.0, dt)
    n = 400
    taus = rng.integers(6, 30, 8)
    errs: list[float] = []
    for tau in taus:
        tr = thin_bed_trace(0.4, int(tau), n, w, t0=50)
        tau_est = tuning_thickness(tr, dt)
        errs.append(abs(tau_est / dt - float(tau)))
    errs_a = np.asarray(errs)
    within = float(np.mean(errs_a <= 1.5))
    return {"synthetic_spectral_decomp": within}
