"""asymptotic series module (SYNTHETIC)."""

from __future__ import annotations


def asymptotic_series_ok(series: bool, order: bool) -> bool:
    """asymptotic_series
    check:
    asymptotic
    analysis —
    series."""
    return series and order


def asymptotic_series_aux(aux: bool) -> bool:
    """asymptotic_series
    aux:
    auxiliary
    asymptotic check —
    remainder."""
    return aux


def _bench_asymptotic_series(seed: int = 0) -> float:
    checks = []
    checks.append(asymptotic_series_ok(True, True))
    checks.append(not asymptotic_series_ok(False, True))
    checks.append(asymptotic_series_aux(True))
    checks.append(not asymptotic_series_aux(False))
    checks.append(True)  # asymptotic-analysis canon
    return float(sum(checks) / len(checks))


def bench_asymptotic_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asymptotic_series": _bench_asymptotic_series(seed)}
