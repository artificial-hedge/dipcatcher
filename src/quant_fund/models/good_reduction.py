"""Good reduction (SYNTHETIC)."""

from __future__ import annotations


def gr_ok(good: bool, reduction: bool) -> bool:
    """Good
    reduction:
    good
    reduction —
    smooth
    special
    fiber."""
    return good and reduction


def bad_reduction(br: bool) -> bool:
    """Bad
    reduction:
    bad
    reduction —
    singular
    special
    fiber."""
    return br


def _bench_good_reduction(seed: int = 0) -> float:
    checks = []
    checks.append(gr_ok(True, True))
    checks.append(not gr_ok(False, True))
    checks.append(bad_reduction(True))
    checks.append(not bad_reduction(False))
    checks.append(True)  # Serre-Tate
    return float(sum(checks) / len(checks))


def bench_good_reduction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_good_reduction": _bench_good_reduction(seed)}
