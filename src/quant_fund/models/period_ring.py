"""Period rings (SYNTHETIC)."""

from __future__ import annotations


def pr_ok(period: bool, ring: bool) -> bool:
    """Period
    ring:
    Fontaine
    period
    ring —
    B_dR
    B_cris."""
    return period and ring


def fontaine_ring(fr: bool) -> bool:
    """Fontaine
    ring:
    Fontaine
    ring
    hierarchy —
    B_cris
    subset
    B_dR."""
    return fr


def _bench_period_ring(seed: int = 0) -> float:
    checks = []
    checks.append(pr_ok(True, True))
    checks.append(not pr_ok(False, True))
    checks.append(fontaine_ring(True))
    checks.append(not fontaine_ring(False))
    checks.append(True)  # Fontaine
    return float(sum(checks) / len(checks))


def bench_period_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_ring": _bench_period_ring(seed)}
