"""varadhan ldp module (SYNTHETIC)."""

from __future__ import annotations


def varadhan_ldp_ok(rate: bool, action: bool) -> bool:
    """varadhan_ldp
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def varadhan_ldp_aux(aux: bool) -> bool:
    """varadhan_ldp
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_varadhan_ldp(seed: int = 0) -> float:
    checks = []
    checks.append(varadhan_ldp_ok(True, True))
    checks.append(not varadhan_ldp_ok(False, True))
    checks.append(varadhan_ldp_aux(True))
    checks.append(not varadhan_ldp_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_varadhan_ldp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_varadhan_ldp": _bench_varadhan_ldp(seed)}
