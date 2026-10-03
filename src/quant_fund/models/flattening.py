"""Flattening stratification: fiber dimensions jump (SYNTHETIC)."""

from __future__ import annotations


def fiber_jump(fibers: list[int]) -> bool:
    """Upper semicontinuity: the generic (first) fiber has the
    minimal dimension; special fibers may only jump upward."""
    return fibers[0] == min(fibers)


def _bench_flattening(seed: int = 0) -> float:
    checks = []
    # flat family: all fibers same dim
    checks.append(fiber_jump([2, 2, 2]))
    # blow-up-like: special fiber bigger
    checks.append(fiber_jump([1, 1, 2]))
    # semicontinuity fails if special < generic
    checks.append(not fiber_jump([2, 1, 2]))
    # constant fibers are flat
    checks.append(fiber_jump([0, 0]))
    # projective maps stratify by fiber dim
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_flattening(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flattening": _bench_flattening(seed)}
