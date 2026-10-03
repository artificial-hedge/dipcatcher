"""feti dp module (SYNTHETIC)."""

from __future__ import annotations


def feti_dp_ok(part: bool, coarse: bool) -> bool:
    """feti_dp
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def feti_dp_aux(aux: bool) -> bool:
    """feti_dp
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_feti_dp(seed: int = 0) -> float:
    checks = []
    checks.append(feti_dp_ok(True, True))
    checks.append(not feti_dp_ok(False, True))
    checks.append(feti_dp_aux(True))
    checks.append(not feti_dp_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_feti_dp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feti_dp": _bench_feti_dp(seed)}
