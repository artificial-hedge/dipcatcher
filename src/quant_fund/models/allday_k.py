"""Allday K-theory (SYNTHETIC)."""

from __future__ import annotations


def ak_ok(allday: bool, k: bool) -> bool:
    """Allday
    K:
    Allday
    K
    theory —
    rational."""
    return allday and k


def rational_k(rk: bool) -> bool:
    """Rational
    K:
    rational
    K
    theory —
    localization."""
    return rk


def _bench_allday_k(seed: int = 0) -> float:
    checks = []
    checks.append(ak_ok(True, True))
    checks.append(not ak_ok(False, True))
    checks.append(rational_k(True))
    checks.append(not rational_k(False))
    checks.append(True)  # Allday
    return float(sum(checks) / len(checks))


def bench_allday_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_allday_k": _bench_allday_k(seed)}
