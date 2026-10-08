"""SIMD-style chunked filter: process rows in vector-width blocks (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 735


def simd_count(arr: np.ndarray, thresh: float, width: int = 8) -> int:
    """Count arr > thresh; process width lanes per step (vectorized)."""
    n = len(arr)
    pad = (-n) % width
    a = np.concatenate([arr, np.full(pad, -np.inf)])
    blocks = a.reshape(-1, width)
    per_block = (blocks > thresh).sum(axis=1)
    return int(per_block.sum())


def bench_simd_filter(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 50
    for _ in range(trials):
        arr = rng.normal(size=int(rng.randint(1, 200)))
        t = float(rng.normal())
        for w in (4, 8, 16):
            ok += float(simd_count(arr, t, w) == int((arr > t).sum()))
    return {"synthetic_simd_exact": ok / (trials * 3)}
