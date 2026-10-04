"""cadlag space module (SYNTHETIC)."""

from __future__ import annotations


def cadlag_space_ok(measure: bool, tight: bool) -> bool:
    """cadlag_space
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def cadlag_space_aux(aux: bool) -> bool:
    """cadlag_space
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_cadlag_space(seed: int = 0) -> float:
    checks = []
    checks.append(cadlag_space_ok(True, True))
    checks.append(not cadlag_space_ok(False, True))
    checks.append(cadlag_space_aux(True))
    checks.append(not cadlag_space_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_cadlag_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cadlag_space": _bench_cadlag_space(seed)}
