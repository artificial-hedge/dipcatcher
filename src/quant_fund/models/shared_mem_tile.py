"""Tiled shared-memory matmul vs numpy oracle (computes, not perf-claims) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 691


def tiled_matmul(a: np.ndarray, b: np.ndarray, tile: int = 8) -> np.ndarray:
    n, m, k = a.shape[0], b.shape[1], a.shape[1]
    c = np.zeros((n, m))
    for i0 in range(0, n, tile):
        for j0 in range(0, m, tile):
            acc = np.zeros((min(tile, n - i0), min(tile, m - j0)))
            for t0 in range(0, k, tile):
                sa = a[i0 : i0 + acc.shape[0], t0 : min(t0 + tile, k)]
                sb = b[t0 : min(t0 + tile, k), j0 : j0 + acc.shape[1]]
                acc += sa @ sb
            c[i0 : i0 + acc.shape[0], j0 : j0 + acc.shape[1]] = acc
    return c


def bench_shared_mem_tile(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 25
    for _ in range(trials):
        n, m, k = int(rng.randint(4, 20)), int(rng.randint(4, 20)), int(rng.randint(4, 20))
        a = rng.normal(0, 1, (n, k))
        b = rng.normal(0, 1, (k, m))
        c = tiled_matmul(a, b, int(rng.choice([4, 8])))
        ok += float(np.allclose(c, a @ b, atol=1e-9))
    return {"synthetic_tiled_exact": ok / trials}
