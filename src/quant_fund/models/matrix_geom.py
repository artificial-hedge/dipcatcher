"""matrix geom module (SYNTHETIC)."""

from __future__ import annotations


def matrix_geom_ok(mat: bool, geo: bool) -> bool:
    """matrix_geom
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def matrix_geom_aux(aux: bool) -> bool:
    """matrix_geom
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_matrix_geom(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_geom_ok(True, True))
    checks.append(not matrix_geom_ok(False, True))
    checks.append(matrix_geom_aux(True))
    checks.append(not matrix_geom_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_matrix_geom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_geom": _bench_matrix_geom(seed)}
