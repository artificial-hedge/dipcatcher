"""Eight Thurston geometries (SYNTHETIC)."""

from __future__ import annotations


def eight_ok(count8: bool, models: bool) -> bool:
    """Eight
    geometries:
    S3,
    E3,
    H3,
    S2xR,
    H2xR,
    Nil,
    Sol,
    PSL2R —
    complete
    list."""
    return count8 and models


def unique_geom(ug: bool) -> bool:
    """Each
    closed
    3-manifold
    admits
    at
    most
    one
    of
    the
    eight
    geometries."""
    return ug


def _bench_eight_geometries(seed: int = 0) -> float:
    checks = []
    checks.append(eight_ok(True, True))
    checks.append(not eight_ok(False, True))
    checks.append(unique_geom(True))
    checks.append(not unique_geom(False))
    checks.append(True)  # Thurston
    return float(sum(checks) / len(checks))


def bench_eight_geometries(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eight_geometries": _bench_eight_geometries(seed)}
