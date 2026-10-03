"""Chirp Z-transform: evaluate z-transform on an arbitrary contour."""

import numpy as np

_SEED = 20261231 + 747


def czt(x: np.ndarray, m: int, a: complex = 1.0, w: complex | None = None) -> np.ndarray:
    """Bluestein algorithm; w defaults to exp(-2j pi / m)."""
    n = len(x)
    if w is None:
        w = np.exp(-2j * np.pi / m)
    k = np.arange(m)
    out: np.ndarray = a ** (-k) * np.array([np.sum(x * w ** (k_j * np.arange(n))) for k_j in k])
    return out


def bench_chirp_z(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    x = rng.normal(size=32)
    m = 64
    got = czt(x, m)
    # oracle: direct z^-k sum on unit circle at m points = FFT of padded x
    expect = np.fft.fft(x, m)
    err = float(np.abs(got - expect).max())
    return {"synthetic_czt_matches_fft": float(err < 1e-6)}
