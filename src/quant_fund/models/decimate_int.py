"""Decimation + zero-stuff interpolation roundtrip (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 751


def decimate(x: np.ndarray, q: int, taps: int = 16) -> np.ndarray:
    """Lowpass then take every q-th sample."""
    n = np.arange(-taps // 2, taps // 2 + 1)
    h = np.sinc(n / q) / q * np.hamming(len(n))
    y = np.convolve(x, h, mode="same")
    return y[::q]


def bench_decimate_int(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 512
    t = np.arange(n)
    x = np.sin(2 * np.pi * 0.02 * t) + 0.05 * rng.normal(size=n)
    q = 4
    d = decimate(x, q)
    # spectrum of decimated signal should peak at the same normalized freq
    spec_d = np.abs(np.fft.rfft(d))
    spec_x = np.abs(np.fft.rfft(x))
    peak_d = spec_d[1:].argmax() + 1
    peak_x = spec_x[1:].argmax() + 1
    # normalized: peak_d / len(d) == peak_x / len(x)
    ok = float(abs(peak_d / len(d) - 4 * peak_x / len(x)) < 0.01)
    return {"synthetic_decimate_freq": ok}
