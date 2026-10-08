"""Cubic Bezier curves (wave 283) (SYNTHETIC).

B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3. Oracles: endpoint
interpolation, convex-hull containment, midpoint symmetry.
"""

import numpy as np

_SEED = 20261231 + 786


def bezier(pts: np.ndarray, t: float) -> np.ndarray:
    p0, p1, p2, p3 = pts
    out: np.ndarray = (
        (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    )
    return out


def bench_bezier_curve(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(6):
        pts = rng.uniform(-2, 2, (4, 2))
        ok += int(np.linalg.norm(bezier(pts, 0.0) - pts[0]) < 1e-12)
        ok += int(np.linalg.norm(bezier(pts, 1.0) - pts[3]) < 1e-12)
        lo, hi = pts.min(0) - 1e-9, pts.max(0) + 1e-9
        mid = np.array([bezier(pts, t / 20.0) for t in range(21)])
        ok += int(((mid >= lo) & (mid <= hi)).all())
        sym = (
            np.linalg.norm(bezier(pts, 0.5) - (pts[0] + 6 * (pts[1] + pts[2]) / 2 + pts[3]) / 8)
            < 1e-9
        )
        ok += int(sym)
    return {"synthetic_bezier": float(ok == 24)}
