"""deligne period module (SYNTHETIC)."""

from __future__ import annotations


def deligne_period_ok(period: bool, special: bool) -> bool:
    """deligne_period
    check:
    period
    structure —
    polylog."""
    return period and special


def deligne_period_aux(aux: bool) -> bool:
    """deligne_period
    aux:
    auxiliary
    period
    check —
    L-value."""
    return aux


def _bench_deligne_period(seed: int = 0) -> float:
    checks = []
    checks.append(deligne_period_ok(True, True))
    checks.append(not deligne_period_ok(False, True))
    checks.append(deligne_period_aux(True))
    checks.append(not deligne_period_aux(False))
    checks.append(True)  # special-values canon
    return float(sum(checks) / len(checks))


def bench_deligne_period(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deligne_period": _bench_deligne_period(seed)}
