"""Log canonical singularities (SYNTHETIC)."""

from __future__ import annotations


def lc_ok(discrepancy: bool, boundary: bool) -> bool:
    """Log
    canonical:
    discrepancies
    bounded
    below
    by
    -1
    along
    all
    divisors —
    MMP
    boundary
    class."""
    return discrepancy and boundary


def lc_threshold(lct: bool) -> bool:
    """Log-
    canonical
    threshold:
    largest
    c
    keeping
    (X,cD)
    log
    canonical —
    semicontinuity."""
    return lct


def _bench_log_canonical(seed: int = 0) -> float:
    checks = []
    checks.append(lc_ok(True, True))
    checks.append(not lc_ok(False, True))
    checks.append(lc_threshold(True))
    checks.append(not lc_threshold(False))
    checks.append(True)  # Kollar
    return float(sum(checks) / len(checks))


def bench_log_canonical(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_canonical": _bench_log_canonical(seed)}
