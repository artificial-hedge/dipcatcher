"""Transversality of curve intersections: independent tangents (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _curve(fn, t: np.ndarray) -> np.ndarray:
    return np.stack([fn(ti) for ti in t])


def _tangent(fn, t: float, h: float = 1e-5) -> np.ndarray:
    return np.asarray((fn(t + h) - fn(t - h)) / (2 * h))


def _intersect_transverse(p1, t1, p2, t2, tol: float = 1e-6) -> bool:
    """At intersection point, tangents must be linearly independent."""
    det = p1[0] * p2[1] - p1[1] * p2[0]
    return bool(abs(det) > tol)


def _bench_transversality(seed: int = 0) -> float:
    checks = []
    # unit circle vs horizontal line y=1/2: two transverse crossings
    circle = lambda t: np.array([np.cos(t), np.sin(t)])  # noqa: E731
    line = lambda t: np.array([t, 0.5])  # noqa: E731
    ts = np.linspace(0, 2 * np.pi, 2001)
    cpts = _curve(circle, ts)
    # find crossings: circle pts where y crosses 0.5
    hits = [i for i in range(len(ts) - 1) if (cpts[i, 1] - 0.5) * (cpts[i + 1, 1] - 0.5) < 0]
    checks.append(len(hits) == 2)
    # verify independence at each crossing
    ok = all(
        _intersect_transverse(_tangent(circle, ts[i]), ts[i], _tangent(line, 0.0), 0.0)
        for i in hits
    )
    checks.append(ok)
    # tangent line y=1 at top of circle: NOT transverse (parallel tangents)
    tl = lambda t: np.array([t, 1.0])  # noqa: E731
    checks.append(
        not _intersect_transverse(_tangent(circle, np.pi / 2), 0.0, _tangent(tl, 0.0), 0.0)
    )
    # two circles crossing at right angles: x^2+y^2=1 and (x-1)^2+y^2=1
    # intersect at (1/2, +-sqrt3/2) with angle 60 deg -> transverse
    c2 = lambda t: np.array([1 + np.cos(t), np.sin(t)])  # noqa: E731
    # find t on circle1 for point (0.5, sqrt3/2): t=pi/3; on circle2: t=2pi/3
    ok2 = _intersect_transverse(_tangent(circle, np.pi / 3), 0.0, _tangent(c2, 2 * np.pi / 3), 0.0)
    checks.append(ok2)
    # osculating curves: y=x^2 vs y=0 at 0 -> tangents parallel -> not transverse
    par = lambda t: np.array([t, t * t])  # noqa: E731
    xax = lambda t: np.array([t, 0.0])  # noqa: E731
    checks.append(not _intersect_transverse(_tangent(par, 0.0), 0.0, _tangent(xax, 0.0), 0.0))
    return float(sum(checks) / len(checks))


def bench_transversality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transversality": _bench_transversality(seed)}
