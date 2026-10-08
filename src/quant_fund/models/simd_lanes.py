"""SIMD-style lane-masked vector kernels vs scalar oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 643


def simd_dot(a: np.ndarray, b: np.ndarray, lanes: int = 4) -> float:
    n = len(a)
    acc = np.zeros(lanes)
    for i in range(0, n - n % lanes, lanes):
        acc += a[i : i + lanes] * b[i : i + lanes]
    # masked tail
    tail = n % lanes
    if tail:
        mask = np.arange(lanes) < tail
        acc[mask] += a[n - tail :] * b[n - tail :]
    return float(acc.sum())


def bench_simd_lanes(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = rng.randint(1, 60)
        a, b = rng.rand(n), rng.rand(n)
        ok += float(abs(simd_dot(a, b) - float(a @ b)) < 1e-9)
    return {"synthetic_simd_correct": ok / trials}
