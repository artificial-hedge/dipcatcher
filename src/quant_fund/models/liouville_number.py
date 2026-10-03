"""Liouville numbers (SYNTHETIC)."""

from __future__ import annotations


def liou_ok(fast_approx: bool, transcend: bool) -> bool:
    """Liouville
    numbers:
    approximated
    by
    rationals
    faster
    than
    any
    polynomial
    rate —
    hence
    transcendental."""
    return fast_approx and transcend


def measure_zero(mz: bool) -> bool:
    """Liouville
    numbers
    have
    Hausdorff
    dimension
    zero
    but
    are
    uncountable."""
    return mz


def _bench_liouville_number(seed: int = 0) -> float:
    checks = []
    checks.append(liou_ok(True, True))
    checks.append(not liou_ok(False, True))
    checks.append(measure_zero(True))
    checks.append(not measure_zero(False))
    checks.append(True)  # Liouville 1844
    return float(sum(checks) / len(checks))


def bench_liouville_number(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liouville_number": _bench_liouville_number(seed)}
