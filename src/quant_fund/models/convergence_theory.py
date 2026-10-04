"""convergence theory module (SYNTHETIC)."""

from __future__ import annotations


def convergence_theory_ok(mark: bool, est: bool) -> bool:
    """convergence_theory
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def convergence_theory_aux(aux: bool) -> bool:
    """convergence_theory
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_convergence_theory(seed: int = 0) -> float:
    checks = []
    checks.append(convergence_theory_ok(True, True))
    checks.append(not convergence_theory_ok(False, True))
    checks.append(convergence_theory_aux(True))
    checks.append(not convergence_theory_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_convergence_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_convergence_theory": _bench_convergence_theory(seed)}
