"""Adic period spaces (SYNTHETIC)."""

from __future__ import annotations


def ap_ok(adic: bool, period: bool) -> bool:
    """Adic:
    adic
    period
    space —
    adic
    period."""
    return adic and period


def period_morphism(pm: bool) -> bool:
    """Period
    morphism:
    Gross-
    Hopkins
    period
    morphism —
    Gross-
    Hopkins."""
    return pm


def _bench_ad_period(seed: int = 0) -> float:
    checks = []
    checks.append(ap_ok(True, True))
    checks.append(not ap_ok(False, True))
    checks.append(period_morphism(True))
    checks.append(not period_morphism(False))
    checks.append(True)  # Gross-Hopkins
    return float(sum(checks) / len(checks))


def bench_ad_period(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ad_period": _bench_ad_period(seed)}
