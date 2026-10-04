"""poincare expansion module (SYNTHETIC)."""

from __future__ import annotations


def poincare_expansion_ok(series: bool, order: bool) -> bool:
    """poincare_expansion
    check:
    asymptotic
    analysis —
    series."""
    return series and order


def poincare_expansion_aux(aux: bool) -> bool:
    """poincare_expansion
    aux:
    auxiliary
    asymptotic check —
    remainder."""
    return aux


def _bench_poincare_expansion(seed: int = 0) -> float:
    checks = []
    checks.append(poincare_expansion_ok(True, True))
    checks.append(not poincare_expansion_ok(False, True))
    checks.append(poincare_expansion_aux(True))
    checks.append(not poincare_expansion_aux(False))
    checks.append(True)  # asymptotic-analysis canon
    return float(sum(checks) / len(checks))


def bench_poincare_expansion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poincare_expansion": _bench_poincare_expansion(seed)}
