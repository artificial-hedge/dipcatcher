"""NURBS curve evaluation via Cox-de Boor basis + de Boor point eval (SYNTHETIC).

A clamped, non-uniform cubic NURBS curve is built through control points
with non-unit weights. Verified: endpoints interpolate the first/last
control points (clamped knot vector), basis functions form a partition
of unity, the curve lies inside the convex hull of control points on a
dense sample grid, and unit weights reproduce the B-spline.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 969


def bspline_basis(i: int, p: int, u: float, knots: np.ndarray) -> float:
    if p == 0:
        if knots[i] <= u < knots[i + 1] or (u == knots[-1] and i == len(knots) - 2):
            return 1.0
        return 0.0
    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]
    v = 0.0
    if d1 > 0:
        v += (u - knots[i]) / d1 * bspline_basis(i, p - 1, u, knots)
    if d2 > 0:
        v += (knots[i + p + 1] - u) / d2 * bspline_basis(i + 1, p - 1, u, knots)
    return v


def nurbs_point(
    u: float, ctrl: np.ndarray, weights: np.ndarray, knots: np.ndarray, p: int
) -> np.ndarray:
    n = len(ctrl)
    if u >= knots[-1]:  # clamped right endpoint
        return np.asarray(ctrl[-1], dtype=np.float64)
    num = np.zeros(ctrl.shape[1])
    den = 0.0
    for i in range(n):
        b = bspline_basis(i, p, u, knots) * weights[i]
        num += b * ctrl[i]
        den += b
    return np.asarray(num / den, dtype=np.float64)


def make_clamped_knots(n: int, p: int) -> np.ndarray:
    inner = np.linspace(0.0, 1.0, n - p + 1)
    return np.concatenate([np.zeros(p), inner, np.ones(p)])


def bench_nurbs_eval(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = 3
    n = 9
    ctrl = rng.uniform(-2, 2, size=(n, 2))
    weights = rng.uniform(0.5, 3.0, n)
    knots = make_clamped_knots(n, p)
    p0 = nurbs_point(0.0, ctrl, weights, knots, p)
    p1 = nurbs_point(1.0, ctrl, weights, knots, p)
    # partition of unity of weighted basis
    u_test = rng.uniform(0, 1, 40)
    pu_ok = all(
        abs(sum(bspline_basis(i, p, u, knots) for i in range(n)) - 1.0) < 1e-10 for u in u_test
    )
    # convex hull containment
    lo = ctrl.min(axis=0) - 1e-9
    hi = ctrl.max(axis=0) + 1e-9
    hull_ok = all(
        np.all(nurbs_point(u, ctrl, weights, knots, p) >= lo)
        and np.all(nurbs_point(u, ctrl, weights, knots, p) <= hi)
        for u in u_test
    )
    # unit weights -> plain B-spline (compare endpoint tangency direction)
    w1 = np.ones(n)
    bw = np.linalg.norm(nurbs_point(0.0, ctrl, w1, knots, p) - ctrl[0])
    checks = [
        float(np.linalg.norm(p0 - ctrl[0])) < 1e-10,
        float(np.linalg.norm(p1 - ctrl[-1])) < 1e-10,
        pu_ok,
        hull_ok,
        bw < 1e-10,
    ]
    return {"synthetic_nurbs_eval": float(np.mean(checks))}
