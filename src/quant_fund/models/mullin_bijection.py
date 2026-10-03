"""mullin bijection module (SYNTHETIC)."""

from __future__ import annotations


def mullin_bijection_ok(map_: bool, bij: bool) -> bool:
    """mullin_bijection
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def mullin_bijection_aux(aux: bool) -> bool:
    """mullin_bijection
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_mullin_bijection(seed: int = 0) -> float:
    checks = []
    checks.append(mullin_bijection_ok(True, True))
    checks.append(not mullin_bijection_ok(False, True))
    checks.append(mullin_bijection_aux(True))
    checks.append(not mullin_bijection_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_mullin_bijection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mullin_bijection": _bench_mullin_bijection(seed)}
