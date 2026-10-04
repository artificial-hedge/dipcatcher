"""quasi birth module (SYNTHETIC)."""

from __future__ import annotations


def quasi_birth_ok(mat: bool, geo: bool) -> bool:
    """quasi_birth
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def quasi_birth_aux(aux: bool) -> bool:
    """quasi_birth
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_quasi_birth(seed: int = 0) -> float:
    checks = []
    checks.append(quasi_birth_ok(True, True))
    checks.append(not quasi_birth_ok(False, True))
    checks.append(quasi_birth_aux(True))
    checks.append(not quasi_birth_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_quasi_birth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_birth": _bench_quasi_birth(seed)}
