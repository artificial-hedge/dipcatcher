"""Mip-mapped texture sampling: trilinear between two levels (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 682


def _mip(tex: np.ndarray, level: int) -> np.ndarray:
    for _ in range(level):
        tex = 0.25 * (tex[0::2, 0::2] + tex[1::2, 0::2] + tex[0::2, 1::2] + tex[1::2, 1::2])
    return tex


def trilinear(tex: np.ndarray, u: float, vv: float, lod: float) -> float:
    l0 = int(np.floor(lod))
    l1 = l0 + 1
    f = lod - l0
    m0 = _mip(tex, l0)
    m1 = _mip(tex, l1)

    def bilin(m: np.ndarray) -> float:
        h, w = m.shape
        x = u * (w - 1)
        y = vv * (h - 1)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        return float(
            m[y0, x0] * (1 - fx) * (1 - fy)
            + m[y0, x1] * fx * (1 - fy)
            + m[y1, x0] * (1 - fx) * fy
            + m[y1, x1] * fx * fy
        )

    return (1 - f) * bilin(m0) + f * bilin(m1)


def bench_mipmap_sample(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        tex = rng.rand(16, 16)
        v = trilinear(tex, rng.rand(), rng.rand(), rng.rand() * 3)
        ok += float(0.0 <= v <= 1.0)
    return {"synthetic_trilinear_range": ok / trials}
