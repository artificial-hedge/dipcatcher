"""Bump mapping: perturbed normals via finite-difference height field."""

import numpy as np

_SEED = 20261231 + 684


def perturb_normal(height: np.ndarray, u: float, vv: float, scale: float = 1.0) -> np.ndarray:
    h, w = height.shape
    x = u * (w - 1)
    y = vv * (h - 1)
    x0, y0 = int(min(x, w - 2)), int(min(y, h - 2))
    dx = (height[y0, x0 + 1] - height[y0, x0]) * scale
    dy = (height[y0 + 1, x0] - height[y0, x0]) * scale
    n = np.array([-dx, -dy, 1.0])
    out: np.ndarray = n / (np.linalg.norm(n) + 1e-12)
    return out


def bench_bump_map(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        hgt = rng.rand(12, 12)
        n = perturb_normal(hgt, rng.rand(), rng.rand(), 2.0)
        ok += float(abs(np.linalg.norm(n) - 1.0) < 1e-9)
        # flat region -> up normal
        flat = np.zeros((8, 8))
        nf = perturb_normal(flat, 0.5, 0.5)
        ok += float(np.allclose(nf, [0, 0, 1]))
    return {"synthetic_bump_unit": ok / (2 * trials)}
