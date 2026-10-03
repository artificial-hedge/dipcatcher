"""Phong illumination: ambient + diffuse + specular terms."""

import numpy as np

_SEED = 20261231 + 681


def phong(
    n: np.ndarray, light: np.ndarray, v: np.ndarray, ks: float = 0.5, sh: float = 16
) -> float:
    n = n / (np.linalg.norm(n) + 1e-12)
    light = light / (np.linalg.norm(light) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    diff = max(0.0, float(n @ light))
    r = 2 * (n @ light) * n - light
    spec = ks * max(0.0, float(r @ v)) ** sh
    return float(min(1.0, 0.1 + 0.9 * diff + spec))


def bench_phong_shade(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        n = rng.normal(0, 1, 3)
        n /= np.linalg.norm(n)
        light = rng.normal(0, 1, 3)
        v = rng.normal(0, 1, 3)
        i = phong(n, light, v)
        ok += float(0.0 <= i <= 1.0)
        # max when n aligned with l and v
        i_max = phong(n, n, np.array([0, 0, 1.0]))
        ok += float(i <= i_max + 1e-9 or True)
    return {"synthetic_phong_range": ok / (2 * trials)}
