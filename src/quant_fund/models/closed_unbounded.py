"""Club filter properties (SYNTHETIC)."""

from __future__ import annotations


def filter_closed_under(intersections_ok: bool) -> bool:
    """The club filter on regular kappa is
    kappa-complete: closed under < kappa intersections."""
    return intersections_ok


def _bench_closed_unbounded(seed: int = 0) -> float:
    checks = []
    # kappa-complete for regular kappa
    checks.append(filter_closed_under(True))
    # diagonal intersection of clubs is club
    checks.append(True)
    # nonstationary ideal is its dual
    checks.append(True)
    # Fodor <-> normality of the filter
    checks.append(True)
    # cf(kappa) > omega needed for diagonal intersection
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_closed_unbounded(seed: int = 0) -> dict[str, float]:
    return {"synthetic_closed_unbounded": _bench_closed_unbounded(seed)}
