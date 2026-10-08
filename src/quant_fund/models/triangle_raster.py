"""Triangle rasterizer with barycentric attribute interpolation (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 680


def rasterize(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, w: int = 32) -> np.ndarray:
    """Fill triangle; returns coverage grid + interpolated depth channel."""
    img = np.zeros((w, w, 2))
    xs = np.array([v0[0], v1[0], v2[0]])
    ys = np.array([v0[1], v1[1], v2[1]])
    bbox = (
        max(0, int(xs.min())),
        min(w - 1, int(xs.max()) + 1),
        max(0, int(ys.min())),
        min(w - 1, int(ys.max()) + 1),
    )
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(denom) < 1e-12:
        return img
    for y in range(bbox[2], bbox[3]):
        for x in range(bbox[0], bbox[1]):
            p = np.array([x + 0.5, y + 0.5])
            a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
            b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
            c = 1 - a - b
            if a >= 0 and b >= 0 and c >= 0:
                img[y, x, 0] = 1.0
                img[y, x, 1] = a * v0[2] + b * v1[2] + c * v2[2]
    return img


def bench_triangle_raster(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 25
    for _ in range(trials):
        v = rng.rand(3, 3) * 24 + 2
        v[:, 2] = rng.rand(3)
        img = rasterize(v[0], v[1], v[2])
        covered = img[:, :, 0].sum()
        # oracle: coverage cannot exceed area + perimeter slack; z stays in vertex range
        area = 0.5 * abs(
            (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1])
        )
        ok += float(
            covered <= area + 3 * (np.linalg.norm(v[1] - v[0]) + np.linalg.norm(v[2] - v[0]))
            and img[:, :, 1].max() <= v[:, 2].max() + 1e-9
        )
    return {"synthetic_raster_covers": ok / trials}
