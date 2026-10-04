"""Parallel transport of a tangent vector along a latitude circle (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def transport_latitude(theta0: float, phi1: float, steps: int = 2000) -> np.ndarray:
    """Parallel transport v0 = d/dtheta along latitude theta=theta0 from phi=0 to phi1.

    In the orthonormal frame {e_theta, e_phi}, the transported vector rotates by
    angle -cos(theta0)*dphi per step.
    """
    angle = 0.0
    dphi = phi1 / steps
    v = np.array([1.0, 0.0])  # components in {e_theta, e_phi}
    for _ in range(steps):
        d_ang = -float(np.cos(theta0)) * dphi
        c, s = float(np.cos(d_ang)), float(np.sin(d_ang))
        v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        angle += d_ang
    return np.asarray(v)


def _bench_parallel_transport(seed: int = 0) -> float:
    checks = []
    # transport along equator: vector stays north-pointing (no rotation)
    v = transport_latitude(np.pi / 2, 2 * np.pi)
    checks.append(np.allclose(v, [1.0, 0.0], atol=1e-6))
    # full loop at latitude theta0: holonomy angle = -2pi cos(theta0) mod 2pi
    theta0 = np.pi / 4
    v = transport_latitude(theta0, 2 * np.pi)
    ang = float(np.arctan2(v[1], v[0]))
    expected = (-2 * np.pi * np.cos(theta0)) % (2 * np.pi)
    checks.append(abs((ang - expected + np.pi) % (2 * np.pi) - np.pi) < 1e-4)
    # norm preserved
    checks.append(abs(float(np.linalg.norm(v)) - 1.0) < 1e-9)
    # partial transport: angle after phi = -cos(theta0)*phi
    v2 = transport_latitude(np.pi / 3, np.pi)
    ang2 = float(np.arctan2(v2[1], v2[0]))
    exp2 = -float(np.cos(np.pi / 3)) * np.pi
    checks.append(abs(ang2 - exp2) < 1e-4)
    return float(sum(checks) / len(checks))


def bench_parallel_transport(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parallel_transport": _bench_parallel_transport(seed)}
