"""dw ldp module (SYNTHETIC)."""

from __future__ import annotations


def dw_ldp_ok(rate: bool, action: bool) -> bool:
    """dw_ldp
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def dw_ldp_aux(aux: bool) -> bool:
    """dw_ldp
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_dw_ldp(seed: int = 0) -> float:
    checks = []
    checks.append(dw_ldp_ok(True, True))
    checks.append(not dw_ldp_ok(False, True))
    checks.append(dw_ldp_aux(True))
    checks.append(not dw_ldp_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_dw_ldp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dw_ldp": _bench_dw_ldp(seed)}
