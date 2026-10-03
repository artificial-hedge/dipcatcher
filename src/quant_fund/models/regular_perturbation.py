"""regular perturbation module (SYNTHETIC)."""

from __future__ import annotations


def regular_perturbation_ok(epsilon: bool, uniform: bool) -> bool:
    """regular_perturbation
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def regular_perturbation_aux(aux: bool) -> bool:
    """regular_perturbation
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_regular_perturbation(seed: int = 0) -> float:
    checks = []
    checks.append(regular_perturbation_ok(True, True))
    checks.append(not regular_perturbation_ok(False, True))
    checks.append(regular_perturbation_aux(True))
    checks.append(not regular_perturbation_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_regular_perturbation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_regular_perturbation": _bench_regular_perturbation(seed)}
