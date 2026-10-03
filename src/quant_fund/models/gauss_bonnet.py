"""Discrete Gauss-Bonnet: sum of angle deficits = 2 pi chi (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def angle_deficit_mesh(angles: list[list[float]]) -> float:
    """Total angle defect: sum over vertices of 2pi - (sum of incident angles)."""
    return float(sum(2 * np.pi - sum(a) for a in angles))


def euler_characteristic(v: int, e: int, f: int) -> int:
    return v - e + f


def _bench_gauss_bonnet(seed: int = 0) -> float:
    checks = []
    # cube as 8 vertices each with 3 right angles: deficit = 8*(2pi - 3*pi/2) = 8*pi/2 = 4pi = 2pi*2
    cube_angles = [[np.pi / 2] * 3 for _ in range(8)]
    checks.append(abs(angle_deficit_mesh(cube_angles) - 4 * np.pi) < 1e-9)
    checks.append(euler_characteristic(8, 12, 6) == 2)
    # tetrahedron: 4 vertices * 3 angles of pi/3: deficit = 4*(2pi - pi) = 4pi
    tet_angles = [[np.pi / 3] * 3 for _ in range(4)]
    checks.append(abs(angle_deficit_mesh(tet_angles) - 4 * np.pi) < 1e-9)
    checks.append(euler_characteristic(4, 6, 4) == 2)
    # octahedron: 6 vertices * 4 angles pi/3: 6*(2pi - 4pi/3) = 6*2pi/3 = 4pi
    oct_angles = [[np.pi / 3] * 4 for _ in range(6)]
    checks.append(abs(angle_deficit_mesh(oct_angles) - 4 * np.pi) < 1e-9)
    # torus: chi = 0, e.g. 4x4 quad grid on torus all flat
    checks.append(euler_characteristic(16, 32, 16) == 0)
    return float(sum(checks) / len(checks))


def bench_gauss_bonnet(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_bonnet": _bench_gauss_bonnet(seed)}
