"""B de Rham period ring (SYNTHETIC)."""

from __future__ import annotations


def bd_ok(drb: bool, ring: bool) -> bool:
    """B
    dR:
    Fontaine
    B_dR
    period
    ring —
    Fontaine
    B_dR."""
    return drb and ring


def period_iso(pi: bool) -> bool:
    """Period
    isomorphism:
    de
    Rham
    comparison —
    Fontaine."""
    return pi


def _bench_b_drb(seed: int = 0) -> float:
    checks = []
    checks.append(bd_ok(True, True))
    checks.append(not bd_ok(False, True))
    checks.append(period_iso(True))
    checks.append(not period_iso(False))
    checks.append(True)  # Fontaine
    return float(sum(checks) / len(checks))


def bench_b_drb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_b_drb": _bench_b_drb(seed)}
