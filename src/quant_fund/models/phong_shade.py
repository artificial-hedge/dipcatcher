"""Phong illumination: ambient + diffuse + specular terms (SYNTHETIC)."""

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
        # aligned normal+light outshines a 60%-tilted light (real check:
        # the previous `or True` made this arm vacuous)
        axis = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tilt = n + 0.6 * np.cross(n, axis)
        ok += float(phong(n, n, v) >= phong(n, tilt, v) - 1e-9)
    if ok != 2 * trials:
        raise ValueError("phong shading oracle failed")
    return {"synthetic_phong_range": ok / (2 * trials)}
