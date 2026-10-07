"""Smoothed-particle hydrodynamics (synthetic 2D SPH-lite) (SYNTHETIC).

Poly6 density kernel + spiky pressure gradient + viscosity.
Verified: (i) poly6 kernel integrates to ~1 (partition of unity
on a dense neighbor grid); (ii) rest-density estimate ≈ ρ0 for a
uniform lattice; (iii) pressure gradient on uniform state ≈ 0.
"""

from __future__ import annotations

import math

H = 0.5
H2 = H * H
POLY6 = 4.0 / (math.pi * H**8)
SPIKY = 10.0 / (math.pi * H**5)
VISC = 40.0 / (math.pi * H**5)


def w_poly6(r2: float) -> float:
    if r2 >= H2:
        return 0.0
    return POLY6 * (H2 - r2) ** 3


def grad_spiky(dx: float, dy: float) -> tuple[float, float]:
    r = math.hypot(dx, dy)
    if r >= H or r < 1e-9:
        return 0.0, 0.0
    f = -SPIKY * (H - r) ** 2 / r
    return f * dx, f * dy


def density(i: int, pos: list[list[float]], mass: list[float]) -> float:
    rho = 0.0
    for j in range(len(pos)):
        dx = pos[i][0] - pos[j][0]
        dy = pos[i][1] - pos[j][1]
        rho += mass[j] * w_poly6(dx * dx + dy * dy)
    return rho


def pressure_grad(
    i: int, pos: list[list[float]], mass: list[float], rho: list[float], p: list[float]
) -> tuple[float, float]:
    gx = gy = 0.0
    for j in range(len(pos)):
        if j == i:
            continue
        dx = pos[i][0] - pos[j][0]
        dy = pos[i][1] - pos[j][1]
        gx_j, gy_j = grad_spiky(dx, dy)
        gx += mass[j] * (p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2) * gx_j
        gy += mass[j] * (p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2) * gy_j
    return gx, gy


def bench_sph_fluid(seed: int = 20261231 + 272) -> dict[str, float]:
    # partition of unity: dense grid, kernel sum ≈ ∫W dx ≈ 1
    n_side = 20
    grid = [[i * 0.1, j * 0.1] for i in range(n_side) for j in range(n_side)]
    center = [1.0, 1.0]
    w_sum = 0.0
    for g in grid:
        dx = g[0] - center[0]
        dy = g[1] - center[1]
        w_sum += w_poly6(dx * dx + dy * dy) * 0.01  # dx*dy = 0.01
    unity = abs(w_sum - 1.0) < 0.05
    # uniform lattice density: larger grid so center sees full kernel support
    pos = [[i * 0.15, j * 0.15] for i in range(12) for j in range(12)]
    mass = [1.0] * len(pos)
    rho = [density(i, pos, mass) for i in range(len(pos))]
    inner = [i for i in range(len(pos)) if 0.6 < pos[i][0] < 1.2 and 0.6 < pos[i][1] < 1.2]
    rho_var = max(rho[i] for i in inner) / min(rho[i] for i in inner)
    uniform_ok = rho_var < 1.05
    # pressure gradient on uniform state ≈ 0 (kernel-interior particle)
    p = [2.0 * r for r in rho]
    ci = min(inner, key=lambda i: abs(pos[i][0] - 0.9) + abs(pos[i][1] - 0.9))
    gx, gy = pressure_grad(ci, pos, mass, rho, p)
    grad_small = math.hypot(gx, gy) < 0.5
    return {
        "synthetic_kernel_unity": float(unity),
        "synthetic_unity_err": float(abs(w_sum - 1.0)),
        "synthetic_uniform_density": float(uniform_ok),
        "synthetic_rho_var": float(rho_var),
        "synthetic_grad_uniform": float(grad_small),
        "synthetic_grad_mag": float(math.hypot(gx, gy)),
    }
