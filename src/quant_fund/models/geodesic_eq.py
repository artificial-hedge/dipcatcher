"""Geodesic equation verification on the sphere (SYNTHIC)."""

from __future__ import annotations

import numpy as np

from quant_fund.models.connection_form import christoffel_sphere


def geodesic_acceleration(
    state: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Given (theta, phi, dtheta, dphi) return geodesic second derivatives."""
    theta, _, dtheta, dphi = state
    ga = christoffel_sphere(theta)
    d2theta = -ga.get((0, 1, 1), 0.0) * dphi * dphi - 2 * ga.get((0, 0, 1), 0.0) * dtheta * dphi
    d2phi = -2 * ga.get((1, 0, 1), 0.0) * dtheta * dphi
    return float(d2theta), float(d2phi)


def _bench_geodesic_eq(seed: int = 0) -> float:
    checks = []
    # great circle: theta(t) = t, phi = const -> d2 = 0
    a, b = geodesic_acceleration((0.7, 0.3, 1.0, 0.0))
    checks.append(abs(a) < 1e-12 and abs(b) < 1e-12)
    # equator with phi moving: theta=pi/2, dphi=1 -> d2theta = 0 (geodesic)
    a, _b = geodesic_acceleration((np.pi / 2, 0.0, 0.0, 1.0))
    checks.append(abs(a) < 1e-12)
    # latitude off equator: nonzero acceleration (not a geodesic)
    a, _b = geodesic_acceleration((np.pi / 4, 0.0, 0.0, 1.0))
    checks.append(abs(a - 0.5) < 1e-9)  # -Gamma^theta_phiphi = sin cos = 0.5
    # Euler integrate a great circle: stays on unit circle in theta-phi space
    theta, phi, vt, vp = np.pi / 2, 0.0, 0.0, 1.0
    dt = 1e-3
    for _ in range(2000):
        at, ap = geodesic_acceleration((theta, phi, vt, vp))
        vt += at * dt
        vp += ap * dt
        theta += vt * dt
        phi += vp * dt
    checks.append(abs(theta - np.pi / 2) < 1e-6)
    return float(sum(checks) / len(checks))


def bench_geodesic_eq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geodesic_eq": _bench_geodesic_eq(seed)}
