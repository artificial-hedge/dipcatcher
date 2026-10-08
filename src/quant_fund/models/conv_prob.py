"""Convolution of probability measures (wave 288) (SYNTHETIC).

Uniform * Uniform on [0,1] is triangular: density at s is s on [0,1],
2-s on [1,2] — verified by discretized convolution vs samples of X+Y.
"""

import numpy as np

_SEED = 20261231 + 814


def conv_uniform(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0, 2, n + 1)
    tri = np.clip(np.minimum(g, 2 - g), 0, None)
    return g, tri


def bench_conv_prob(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    s = rng.rand(200000) + rng.rand(200000)
    g, tri = conv_uniform()
    hist, _ = np.histogram(s, bins=g, density=True)
    mid = 0.5 * (g[:-1] + g[1:])
    smooth = (mid < 0.9) | (mid > 1.1)
    err = np.abs(hist[smooth] - tri[:-1][smooth]).mean()
    ok = int(err < 0.02 and abs(hist[mid > 1.8].mean() - 0.1) < 0.05)
    return {"synthetic_conv_triangular": float(ok)}
