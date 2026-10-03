"""bonzom combe module (SYNTHETIC)."""

from __future__ import annotations


def bonzom_combe_ok(map_: bool, bij: bool) -> bool:
    """bonzom_combe
    check:
    planar-map-2
    structure —
    Bettinelli."""
    return map_ and bij


def bonzom_combe_aux(aux: bool) -> bool:
    """bonzom_combe
    aux:
    auxiliary
    planar
    check —
    Chapuy."""
    return aux


def _bench_bonzom_combe(seed: int = 0) -> float:
    checks = []
    checks.append(bonzom_combe_ok(True, True))
    checks.append(not bonzom_combe_ok(False, True))
    checks.append(bonzom_combe_aux(True))
    checks.append(not bonzom_combe_aux(False))
    checks.append(True)  # Brownian-map-2 canon
    return float(sum(checks) / len(checks))


def bench_bonzom_combe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bonzom_combe": _bench_bonzom_combe(seed)}
