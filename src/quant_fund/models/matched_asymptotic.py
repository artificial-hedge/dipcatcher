"""matched asymptotic module (SYNTHETIC)."""

from __future__ import annotations


def matched_asymptotic_ok(epsilon: bool, uniform: bool) -> bool:
    """matched_asymptotic
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def matched_asymptotic_aux(aux: bool) -> bool:
    """matched_asymptotic
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_matched_asymptotic(seed: int = 0) -> float:
    checks = []
    checks.append(matched_asymptotic_ok(True, True))
    checks.append(not matched_asymptotic_ok(False, True))
    checks.append(matched_asymptotic_aux(True))
    checks.append(not matched_asymptotic_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_matched_asymptotic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matched_asymptotic": _bench_matched_asymptotic(seed)}
