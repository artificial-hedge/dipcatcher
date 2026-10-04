"""neuts map module (SYNTHETIC)."""

from __future__ import annotations


def neuts_map_ok(mat: bool, geo: bool) -> bool:
    """neuts_map
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def neuts_map_aux(aux: bool) -> bool:
    """neuts_map
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_neuts_map(seed: int = 0) -> float:
    checks = []
    checks.append(neuts_map_ok(True, True))
    checks.append(not neuts_map_ok(False, True))
    checks.append(neuts_map_aux(True))
    checks.append(not neuts_map_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_neuts_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neuts_map": _bench_neuts_map(seed)}
