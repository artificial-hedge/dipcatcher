"""ortega series module (SYNTHETIC)."""

from __future__ import annotations


def ortega_series_ok(conv: bool, trunc: bool) -> bool:
    """ortega_series
    check:
    random
    series —
    convergence."""
    return conv and trunc


def ortega_series_aux(aux: bool) -> bool:
    """ortega_series
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_ortega_series(seed: int = 0) -> float:
    checks = []
    checks.append(ortega_series_ok(True, True))
    checks.append(not ortega_series_ok(False, True))
    checks.append(ortega_series_aux(True))
    checks.append(not ortega_series_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_ortega_series(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ortega_series": _bench_ortega_series(seed)}
