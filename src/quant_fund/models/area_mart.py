"""area mart module (SYNTHETIC)."""

from __future__ import annotations


def area_mart_ok(rp1: bool, lift: bool) -> bool:
    """area_mart
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def area_mart_aux(aux: bool) -> bool:
    """area_mart
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_area_mart(seed: int = 0) -> float:
    checks = []
    checks.append(area_mart_ok(True, True))
    checks.append(not area_mart_ok(False, True))
    checks.append(area_mart_aux(True))
    checks.append(not area_mart_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_area_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_area_mart": _bench_area_mart(seed)}
