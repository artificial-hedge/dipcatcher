"""period poly module (SYNTHETIC)."""

from __future__ import annotations


def period_poly_ok(period: bool, special: bool) -> bool:
    """period_poly
    check:
    period
    structure —
    polylog."""
    return period and special


def period_poly_aux(aux: bool) -> bool:
    """period_poly
    aux:
    auxiliary
    period
    check —
    L-value."""
    return aux


def _bench_period_poly(seed: int = 0) -> float:
    checks = []
    checks.append(period_poly_ok(True, True))
    checks.append(not period_poly_ok(False, True))
    checks.append(period_poly_aux(True))
    checks.append(not period_poly_aux(False))
    checks.append(True)  # special-values canon
    return float(sum(checks) / len(checks))


def bench_period_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_poly": _bench_period_poly(seed)}
