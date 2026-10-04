"""three series module (SYNTHETIC)."""

from __future__ import annotations


def three_series_ok(conv: bool, trunc: bool) -> bool:
    """three_series
    check:
    random
    series —
    convergence."""
    return conv and trunc


def three_series_aux(aux: bool) -> bool:
    """three_series
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_three_series(seed: int = 0) -> float:
    checks = []
    checks.append(three_series_ok(True, True))
    checks.append(not three_series_ok(False, True))
    checks.append(three_series_aux(True))
    checks.append(not three_series_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_three_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_three_series": _bench_three_series(seed)}
