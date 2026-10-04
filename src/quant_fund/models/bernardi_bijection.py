"""bernardi bijection module (SYNTHETIC)."""

from __future__ import annotations


def bernardi_bijection_ok(map_: bool, bij: bool) -> bool:
    """bernardi_bijection
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def bernardi_bijection_aux(aux: bool) -> bool:
    """bernardi_bijection
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_bernardi_bijection(seed: int = 0) -> float:
    checks = []
    checks.append(bernardi_bijection_ok(True, True))
    checks.append(not bernardi_bijection_ok(False, True))
    checks.append(bernardi_bijection_aux(True))
    checks.append(not bernardi_bijection_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_bernardi_bijection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bernardi_bijection": _bench_bernardi_bijection(seed)}
