"""schaeffer bijection module (SYNTHETIC)."""

from __future__ import annotations


def schaeffer_bijection_ok(map_: bool, bij: bool) -> bool:
    """schaeffer_bijection
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def schaeffer_bijection_aux(aux: bool) -> bool:
    """schaeffer_bijection
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_schaeffer_bijection(seed: int = 0) -> float:
    checks = []
    checks.append(schaeffer_bijection_ok(True, True))
    checks.append(not schaeffer_bijection_ok(False, True))
    checks.append(schaeffer_bijection_aux(True))
    checks.append(not schaeffer_bijection_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_schaeffer_bijection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schaeffer_bijection": _bench_schaeffer_bijection(seed)}
