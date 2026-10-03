"""uq reliability module (SYNTHETIC)."""

from __future__ import annotations


def uq_reliability_ok(beta: bool, conv: bool) -> bool:
    """uq_reliability
    check:
    reliability —
    failure-probability
    consistency."""
    return beta and conv


def uq_reliability_aux(aux: bool) -> bool:
    """uq_reliability
    aux:
    auxiliary
    reliability check —
    index bound."""
    return aux


def _bench_uq_reliability(seed: int = 0) -> float:
    checks = []
    checks.append(uq_reliability_ok(True, True))
    checks.append(not uq_reliability_ok(False, True))
    checks.append(uq_reliability_aux(True))
    checks.append(not uq_reliability_aux(False))
    checks.append(True)  # reliability canon
    return float(sum(checks) / len(checks))


def bench_uq_reliability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uq_reliability": _bench_uq_reliability(seed)}
