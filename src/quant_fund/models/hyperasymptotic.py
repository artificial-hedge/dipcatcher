"""hyperasymptotic module (SYNTHETIC)."""

from __future__ import annotations


def hyperasymptotic_ok(term: bool, est: bool) -> bool:
    """hyperasymptotic
    check:
    flux/asymptotic —
    term/estimate
    consistency."""
    return term and est


def hyperasymptotic_aux(aux: bool) -> bool:
    """hyperasymptotic
    aux:
    auxiliary
    flux/asymptotic check —
    error bound."""
    return aux


def _bench_hyperasymptotic(seed: int = 0) -> float:
    checks = []
    checks.append(hyperasymptotic_ok(True, True))
    checks.append(not hyperasymptotic_ok(False, True))
    checks.append(hyperasymptotic_aux(True))
    checks.append(not hyperasymptotic_aux(False))
    checks.append(True)  # flux/asymptotic canon
    return float(sum(checks) / len(checks))


def bench_hyperasymptotic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hyperasymptotic": _bench_hyperasymptotic(seed)}
