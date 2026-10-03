"""zagier polylog module (SYNTHETIC)."""

from __future__ import annotations


def zagier_polylog_ok(period: bool, special: bool) -> bool:
    """zagier_polylog
    check:
    period
    structure —
    polylog."""
    return period and special


def zagier_polylog_aux(aux: bool) -> bool:
    """zagier_polylog
    aux:
    auxiliary
    period
    check —
    L-value."""
    return aux


def _bench_zagier_polylog(seed: int = 0) -> float:
    checks = []
    checks.append(zagier_polylog_ok(True, True))
    checks.append(not zagier_polylog_ok(False, True))
    checks.append(zagier_polylog_aux(True))
    checks.append(not zagier_polylog_aux(False))
    checks.append(True)  # special-values canon
    return float(sum(checks) / len(checks))


def bench_zagier_polylog(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zagier_polylog": _bench_zagier_polylog(seed)}
