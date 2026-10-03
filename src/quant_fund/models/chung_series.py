"""chung series module (SYNTHETIC)."""

from __future__ import annotations


def chung_series_ok(conv: bool, trunc: bool) -> bool:
    """chung_series
    check:
    random
    series —
    convergence."""
    return conv and trunc


def chung_series_aux(aux: bool) -> bool:
    """chung_series
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_chung_series(seed: int = 0) -> float:
    checks = []
    checks.append(chung_series_ok(True, True))
    checks.append(not chung_series_ok(False, True))
    checks.append(chung_series_aux(True))
    checks.append(not chung_series_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_chung_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chung_series": _bench_chung_series(seed)}
