"""SSAO-lite: hemisphere sample occlusion ratio around a depth map."""

import numpy as np

_SEED = 20261231 + 685


def ssao(depth: np.ndarray, rng: np.random.RandomState, n_samples: int = 8) -> np.ndarray:
    h, w = depth.shape
    occ = np.zeros((h, w))
    for _ in range(n_samples):
        dx = rng.randint(-3, 4, (h, w))
        dy = rng.randint(-3, 4, (h, w))
        dz = rng.randint(0, 3, (h, w))
        xs = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
        ys = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
        occ += (depth[ys, xs] < depth - 0.01 * dz).astype(float)
    return 1.0 - occ / n_samples


def bench_ssao_lite(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 25
    for _ in range(trials):
        depth = np.cumsum(rng.rand(20, 20), axis=1)
        ao = ssao(depth, rng)
        ok += float(ao.min() >= 0.0 and ao.max() <= 1.0)
    return {"synthetic_ssao_range": ok / trials}
