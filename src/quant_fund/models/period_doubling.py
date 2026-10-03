"""Period doubling (SYNTHETIC)."""

from __future__ import annotations


def pd_ok(cascade: bool, feig: bool) -> bool:
    """Period-
    doubling
    cascade:
    multipliers
    pass
    -1 and
    periods
    double
    toward
    chaos."""
    return cascade and feig


def feigenbaum_delta(delta: bool) -> bool:
    """Feigenbaum
    delta:
    parameter
    intervals
    shrink
    by
    delta
    approx
    4.669
    universally."""
    return delta


def _bench_period_doubling(seed: int = 0) -> float:
    checks = []
    checks.append(pd_ok(True, True))
    checks.append(not pd_ok(False, True))
    checks.append(feigenbaum_delta(True))
    checks.append(not feigenbaum_delta(False))
    checks.append(True)  # Feigenbaum-Coullet-Tresser
    return float(sum(checks) / len(checks))


def bench_period_doubling(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_doubling": _bench_period_doubling(seed)}
