"""Windowed-sinc FIR design + frequency response check (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 748


def fir_lp(cutoff: float, ntap: int) -> np.ndarray:
    """Ideal lowpass impulse response × Hamming window."""
    n = np.arange(ntap) - (ntap - 1) / 2
    h = np.sinc(2 * cutoff * n)
    out: np.ndarray = h * np.hamming(ntap)
    return out


def freq_response(h: np.ndarray, omega: np.ndarray) -> np.ndarray:
    n = np.arange(len(h))
    return np.array([np.abs((h * np.exp(-1j * w * n)).sum()) for w in omega])


def bench_fir_window(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    del rng
    h = fir_lp(0.2, 63)
    om = np.linspace(0, np.pi, 64)
    resp = freq_response(h, om)
    passband = resp[om < 0.25 * np.pi].min()
    stopband = resp[om > 0.6 * np.pi].max()
    return {"synthetic_fir_shape": float(passband > 0.9 and stopband < 0.1)}
