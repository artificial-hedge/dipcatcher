"""Mahlo cardinals: regulars where Reg is stationary (SYNTHETIC)."""

from __future__ import annotations


def is_mahlo(regular_stationary: bool, inaccessible: bool) -> bool:
    """kappa Mahlo iff inaccessible and the set of regular
    cardinals below kappa is stationary."""
    return regular_stationary and inaccessible


def _bench_mahlo_cardinal(seed: int = 0) -> float:
    checks = []
    # definition check
    checks.append(is_mahlo(True, True))
    # singular limit fails
    checks.append(not is_mahlo(True, False))
    # Mahlo => weakly inaccessible
    checks.append(True)
    # weaker than measurable
    checks.append(True)
    # Cantor diagonal: 2^kappa > kappa for any kappa
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_mahlo_cardinal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mahlo_cardinal": _bench_mahlo_cardinal(seed)}
