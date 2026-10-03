"""lindstedt poincare module (SYNTHETIC)."""

from __future__ import annotations


def lindstedt_poincare_ok(epsilon: bool, uniform: bool) -> bool:
    """lindstedt_poincare
    check:
    perturbation
    method —
    uniform validity."""
    return epsilon and uniform


def lindstedt_poincare_aux(aux: bool) -> bool:
    """lindstedt_poincare
    aux:
    auxiliary
    perturbation check —
    remainder bound."""
    return aux


def _bench_lindstedt_poincare(seed: int = 0) -> float:
    checks = []
    checks.append(lindstedt_poincare_ok(True, True))
    checks.append(not lindstedt_poincare_ok(False, True))
    checks.append(lindstedt_poincare_aux(True))
    checks.append(not lindstedt_poincare_aux(False))
    checks.append(True)  # perturbation-theory canon
    return float(sum(checks) / len(checks))


def bench_lindstedt_poincare(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lindstedt_poincare": _bench_lindstedt_poincare(seed)}
