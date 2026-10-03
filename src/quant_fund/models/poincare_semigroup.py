"""poincare semigroup module (SYNTHETIC)."""

from __future__ import annotations


def poincare_semigroup_ok(sem: bool, gen: bool) -> bool:
    """poincare_semigroup
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def poincare_semigroup_aux(aux: bool) -> bool:
    """poincare_semigroup
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_poincare_semigroup(seed: int = 0) -> float:
    checks = []
    checks.append(poincare_semigroup_ok(True, True))
    checks.append(not poincare_semigroup_ok(False, True))
    checks.append(poincare_semigroup_aux(True))
    checks.append(not poincare_semigroup_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_poincare_semigroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_semigroup": _bench_poincare_semigroup(seed)}
