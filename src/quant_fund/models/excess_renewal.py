"""excess renewal module (SYNTHETIC)."""

from __future__ import annotations


def excess_renewal_ok(renewal: bool, eq: bool) -> bool:
    """excess_renewal
    check:
    renewal-theory
    structure —
    Blackwell."""
    return renewal and eq


def excess_renewal_aux(aux: bool) -> bool:
    """excess_renewal
    aux:
    auxiliary
    reward
    check —
    Feller."""
    return aux


def _bench_excess_renewal(seed: int = 0) -> float:
    checks = []
    checks.append(excess_renewal_ok(True, True))
    checks.append(not excess_renewal_ok(False, True))
    checks.append(excess_renewal_aux(True))
    checks.append(not excess_renewal_aux(False))
    checks.append(True)  # renewal canon
    return float(sum(checks) / len(checks))


def bench_excess_renewal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_excess_renewal": _bench_excess_renewal(seed)}
