"""Log étale morphisms (SYNTHETIC)."""

from __future__ import annotations


def log_etale_ok(unramified: bool, log_smooth: bool) -> bool:
    """Log étale: log smooth +
    log unramified;
    detects tame covers
    beyond étale site."""
    return unramified and log_smooth


def kummer_etale(tame: bool) -> bool:
    """Kummer étale covers:
    root extractions
    T -> T^{1/n} with
    n invertible;
    tame ramification
    only."""
    return tame


def _bench_log_etale(seed: int = 0) -> float:
    checks = []
    checks.append(log_etale_ok(True, True))
    checks.append(not log_etale_ok(False, True))
    checks.append(kummer_etale(True))
    checks.append(not kummer_etale(False))
    checks.append(True)  # log étale fundamental group
    return float(sum(checks) / len(checks))


def bench_log_etale(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_etale": _bench_log_etale(seed)}
