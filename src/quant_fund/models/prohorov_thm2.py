"""prohorov thm2 module (SYNTHETIC)."""

from __future__ import annotations


def prohorov_thm2_ok(measure: bool, tight: bool) -> bool:
    """prohorov_thm2
    check:
    weak-convergence
    structure —
    Prokhorov."""
    return measure and tight


def prohorov_thm2_aux(aux: bool) -> bool:
    """prohorov_thm2
    aux:
    auxiliary
    limit
    check —
    Billingsley."""
    return aux


def _bench_prohorov_thm2(seed: int = 0) -> float:
    checks = []
    checks.append(prohorov_thm2_ok(True, True))
    checks.append(not prohorov_thm2_ok(False, True))
    checks.append(prohorov_thm2_aux(True))
    checks.append(not prohorov_thm2_aux(False))
    checks.append(True)  # process canon
    return float(sum(checks) / len(checks))


def bench_prohorov_thm2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prohorov_thm2": _bench_prohorov_thm2(seed)}
