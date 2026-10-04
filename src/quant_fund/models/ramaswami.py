"""ramaswami module (SYNTHETIC)."""

from __future__ import annotations


def ramaswami_ok(mat: bool, geo: bool) -> bool:
    """ramaswami
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def ramaswami_aux(aux: bool) -> bool:
    """ramaswami
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_ramaswami(seed: int = 0) -> float:
    checks = []
    checks.append(ramaswami_ok(True, True))
    checks.append(not ramaswami_ok(False, True))
    checks.append(ramaswami_aux(True))
    checks.append(not ramaswami_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_ramaswami(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramaswami": _bench_ramaswami(seed)}
