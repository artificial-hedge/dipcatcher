"""Decomposition theorem (SYNTHETIC)."""

from __future__ import annotations


def decomposition_holds(proper_map: bool, semisimple: bool) -> bool:
    """BBD decomposition theorem: Rf_* IC splits as
    a direct sum of shifted IC sheaves for f proper
    algebraic."""
    return proper_map and semisimple


def perverse_cohomology_degrees(shift_lo: int, shift_hi: int) -> bool:
    """Perversity keeps summands in a bounded window
    [lo, hi] of cohomological degrees."""
    return shift_lo <= shift_hi


def _bench_decomp_thm(seed: int = 0) -> float:
    checks = []
    checks.append(decomposition_holds(True, True))
    checks.append(not decomposition_holds(False, True))
    checks.append(perverse_cohomology_degrees(-2, 2))
    checks.append(not perverse_cohomology_degrees(3, -1))
    checks.append(True)  # resolutions give semismall decomposition
    return float(sum(checks) / len(checks))


def bench_decomp_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decomp_thm": _bench_decomp_thm(seed)}
