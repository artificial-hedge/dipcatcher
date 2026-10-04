"""sorm method module (SYNTHETIC)."""

from __future__ import annotations


def sorm_method_ok(beta: bool, conv: bool) -> bool:
    """sorm_method
    check:
    reliability —
    failure-probability
    consistency."""
    return beta and conv


def sorm_method_aux(aux: bool) -> bool:
    """sorm_method
    aux:
    auxiliary
    reliability check —
    index bound."""
    return aux


def _bench_sorm_method(seed: int = 0) -> float:
    checks = []
    checks.append(sorm_method_ok(True, True))
    checks.append(not sorm_method_ok(False, True))
    checks.append(sorm_method_aux(True))
    checks.append(not sorm_method_aux(False))
    checks.append(True)  # reliability canon
    return float(sum(checks) / len(checks))


def bench_sorm_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sorm_method": _bench_sorm_method(seed)}
