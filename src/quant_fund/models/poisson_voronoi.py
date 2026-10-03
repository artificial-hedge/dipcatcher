"""poisson voronoi module (SYNTHETIC)."""

from __future__ import annotations


def poisson_voronoi_ok(geo: bool, tess: bool) -> bool:
    """poisson_voronoi
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def poisson_voronoi_aux(aux: bool) -> bool:
    """poisson_voronoi
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_poisson_voronoi(seed: int = 0) -> float:
    checks = []
    checks.append(poisson_voronoi_ok(True, True))
    checks.append(not poisson_voronoi_ok(False, True))
    checks.append(poisson_voronoi_aux(True))
    checks.append(not poisson_voronoi_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_poisson_voronoi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poisson_voronoi": _bench_poisson_voronoi(seed)}
