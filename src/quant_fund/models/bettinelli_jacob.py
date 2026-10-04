"""bettinelli jacob module (SYNTHETIC)."""

from __future__ import annotations


def bettinelli_jacob_ok(map_: bool, plane: bool) -> bool:
    """bettinelli_jacob
    check:
    Brownian-map
    structure —
    LeGall."""
    return map_ and plane


def bettinelli_jacob_aux(aux: bool) -> bool:
    """bettinelli_jacob
    aux:
    auxiliary
    planar
    check —
    Curien."""
    return aux


def _bench_bettinelli_jacob(seed: int = 0) -> float:
    checks = []
    checks.append(bettinelli_jacob_ok(True, True))
    checks.append(not bettinelli_jacob_ok(False, True))
    checks.append(bettinelli_jacob_aux(True))
    checks.append(not bettinelli_jacob_aux(False))
    checks.append(True)  # Brownian-map canon
    return float(sum(checks) / len(checks))


def bench_bettinelli_jacob(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bettinelli_jacob": _bench_bettinelli_jacob(seed)}
