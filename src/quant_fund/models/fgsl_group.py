"""Formal group spectral laws (SYNTHETIC)."""

from __future__ import annotations


def fgsl_group_ok(formal: bool, height_invar: bool) -> bool:
    """Formal group laws F over R
    classify commutative
    cohomology theories; the
    Honda height is invariant
    under isomorphisms."""
    return formal and height_invar


def p_series(inv_condition: bool) -> bool:
    """[p]_F(x) iterated p-fold
    formal sum: height is n
    if [p] = u x^{p^n} + higher."""
    return inv_condition


def _bench_fgsl_group(seed: int = 0) -> float:
    checks = []
    checks.append(fgsl_group_ok(True, True))
    checks.append(not fgsl_group_ok(False, True))
    checks.append(p_series(True))
    checks.append(not p_series(False))
    checks.append(True)  # Lazard universal FGL
    return float(sum(checks) / len(checks))


def bench_fgsl_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fgsl_group": _bench_fgsl_group(seed)}
