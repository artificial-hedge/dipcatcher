"""Surface area element integration (wave 287).

Area = integral sqrt(det I) dudv over the domain: sphere (4*pi*r^2) and
torus (4*pi^2*R*r) recovered by quadrature.
"""

import numpy as np

_SEED = 20261231 + 811


def _sphere_met(u: float, v: float, r: float = 2.0) -> float:
    return float(r * r * np.sin(u))


def _torus_met(u: float, v: float, r: float = 1.0, big: float = 3.0) -> float:
    return float(r * (big + r * np.cos(u)))


def _quad_area(met, n: int = 700, u_max: float = np.pi) -> float:
    u = np.linspace(1e-4, u_max - 1e-4, n)
    v = np.linspace(0, 2 * np.pi, n)
    du, dv = u[1] - u[0], v[1] - v[0]
    tot = float(sum(met(a, b) for a in u for b in v))
    return float(tot * du * dv)


def bench_surf_area(seed: int = _SEED) -> dict[str, float]:
    a_sph = _quad_area(_sphere_met)
    a_tor = _quad_area(_torus_met, u_max=2 * np.pi)
    want_sph = 4 * np.pi * 4.0
    want_tor = 4 * np.pi**2 * 3.0
    return {
        "synthetic_surf_area": float(
            abs(a_sph - want_sph) / want_sph < 5e-3 and abs(a_tor - want_tor) / want_tor < 5e-3
        )
    }
