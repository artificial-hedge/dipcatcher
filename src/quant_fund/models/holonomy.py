"""Holonomy of latitude loops on the sphere = enclosed solid angle (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def solid_angle_cap(theta0: float) -> float:
    """Solid angle of the polar cap bounded by latitude theta0: 2pi(1-cos theta0)."""
    return float(2 * np.pi * (1.0 - np.cos(theta0)))


def holonomy_angle(theta0: float) -> float:
    """Rotation angle after transporting a vector once around latitude theta0.

    Gauss-Bonnet: holonomy = integral of K dA over enclosed cap = solid angle.
    """
    return solid_angle_cap(theta0) % (2 * np.pi)


def geodesic_curvature_latitude(theta0: float) -> float:
    """kg of the latitude circle (as a curve on the sphere) = cot(theta0)."""
    return float(np.cos(theta0) / np.sin(theta0))


def _bench_holonomy(seed: int = 0) -> float:
    checks = []
    # equator bounds a hemisphere: solid angle 2pi -> holonomy 0 mod 2pi
    checks.append(
        abs(holonomy_angle(np.pi / 2)) < 1e-12 or abs(holonomy_angle(np.pi / 2) - 2 * np.pi) < 1e-12
    )
    # latitude at pi/4: solid angle 2pi(1-sqrt2/2) ~ 1.842
    expected = 2 * np.pi * (1 - np.sqrt(2) / 2)
    checks.append(abs(holonomy_angle(np.pi / 4) - expected) < 1e-9)
    # Gauss-Bonnet on geodesic-polar triangle is consistent: cap solid angle > 0, < 2pi
    checks.append(0.0 < solid_angle_cap(np.pi / 6) < 2 * np.pi)
    # kg latitude circle: equator is a geodesic (kg = 0)
    checks.append(abs(geodesic_curvature_latitude(np.pi / 2)) < 1e-12)
    # kg integrates to 2pi - holonomy over the loop (Gauss-Bonnet for disc)
    theta0 = np.pi / 3
    kg_len = geodesic_curvature_latitude(theta0) * 2 * np.pi * np.sin(theta0)
    checks.append(abs((2 * np.pi - kg_len) - solid_angle_cap(theta0)) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_holonomy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holonomy": _bench_holonomy(seed)}
