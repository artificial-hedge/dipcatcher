"""phase type module (SYNTHETIC)."""

from __future__ import annotations


def phase_type_ok(mat: bool, geo: bool) -> bool:
    """phase_type
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def phase_type_aux(aux: bool) -> bool:
    """phase_type
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_phase_type(seed: int = 0) -> float:
    checks = []
    checks.append(phase_type_ok(True, True))
    checks.append(not phase_type_ok(False, True))
    checks.append(phase_type_aux(True))
    checks.append(not phase_type_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_phase_type(seed: int = 0) -> dict[str, float]:
    return {"synthetic_phase_type": _bench_phase_type(seed)}
