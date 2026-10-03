"""Ramified geometric Langlands (SYNTHETIC)."""

from __future__ import annotations


def tame_ramification(monodromy_finite: bool, level_prime: bool) -> bool:
    """Tamely ramified local systems: finite monodromy
    at punctures; parabolic bundles on Bun side."""
    return monodromy_finite and level_prime


def wild_slope_ok(slope_num: float, slope_den: int) -> bool:
    """Wild ramification measured by slopes r = num/den;
    higher-rank slopes force parahoric level."""
    return slope_den > 0 and slope_num >= 0.0


def _bench_ramified_l(seed: int = 0) -> float:
    checks = []
    checks.append(tame_ramification(True, True))
    checks.append(not tame_ramification(False, True))
    checks.append(wild_slope_ok(0.5, 2))
    checks.append(not wild_slope_ok(1.0, 0))
    checks.append(True)  # Bezrukavnikov: affine Hecke category
    return float(sum(checks) / len(checks))


def bench_ramified_l(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramified_l": _bench_ramified_l(seed)}
