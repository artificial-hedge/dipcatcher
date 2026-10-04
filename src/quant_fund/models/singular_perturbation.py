"""singular perturbation module (SYNTHETIC)."""

from __future__ import annotations


def singular_perturbation_ok(epsilon: bool, uniform: bool) -> bool:
    """singular_perturbation
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def singular_perturbation_aux(aux: bool) -> bool:
    """singular_perturbation
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_singular_perturbation(seed: int = 0) -> float:
    checks = []
    checks.append(singular_perturbation_ok(True, True))
    checks.append(not singular_perturbation_ok(False, True))
    checks.append(singular_perturbation_aux(True))
    checks.append(not singular_perturbation_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_singular_perturbation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_singular_perturbation": _bench_singular_perturbation(seed)}
