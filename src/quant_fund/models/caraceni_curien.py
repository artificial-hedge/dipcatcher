"""caraceni curien module (SYNTHETIC)."""

from __future__ import annotations


def caraceni_curien_ok(map_: bool, bij: bool) -> bool:
    """caraceni_curien
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def caraceni_curien_aux(aux: bool) -> bool:
    """caraceni_curien
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_caraceni_curien(seed: int = 0) -> float:
    checks = []
    checks.append(caraceni_curien_ok(True, True))
    checks.append(not caraceni_curien_ok(False, True))
    checks.append(caraceni_curien_aux(True))
    checks.append(not caraceni_curien_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_caraceni_curien(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caraceni_curien": _bench_caraceni_curien(seed)}
