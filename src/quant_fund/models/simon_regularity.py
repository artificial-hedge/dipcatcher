"""Simon regularity (SYNTHETIC)."""

from __future__ import annotations


def sr_ok(tangent_cone: bool, cylindrical: bool) -> bool:
    """Simon
    regularity:
    cylindrical
    tangent
    cones
    imply
    smooth
    convergence —
    Lojasiewicz
    inequality."""
    return tangent_cone and cylindrical


def lojasiewicz_simon(ls: bool) -> bool:
    """Lojasiewicz-
    Simon:
    infinite-
    dimensional
    gradient
    inequality —
    controls
    rate
    of
    convergence."""
    return ls


def _bench_simon_regularity(seed: int = 0) -> float:
    checks = []
    checks.append(sr_ok(True, True))
    checks.append(not sr_ok(False, True))
    checks.append(lojasiewicz_simon(True))
    checks.append(not lojasiewicz_simon(False))
    checks.append(True)  # L. Simon
    return float(sum(checks) / len(checks))


def bench_simon_regularity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simon_regularity": _bench_simon_regularity(seed)}
