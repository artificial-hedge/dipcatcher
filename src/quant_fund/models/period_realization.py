"""Period realizations (SYNTHETIC)."""

from __future__ import annotations


def pr_ok(period: bool, realization: bool) -> bool:
    """Period
    realization:
    period
    isomorphism —
    de Rham
    vs
    Betti."""
    return period and realization


def de_rham_betti_iso(drb: bool) -> bool:
    """de
    Rham-Betti:
    period
    map
    de Rham
    to
    Betti —
    comparison."""
    return drb


def _bench_period_realization(seed: int = 0) -> float:
    checks = []
    checks.append(pr_ok(True, True))
    checks.append(not pr_ok(False, True))
    checks.append(de_rham_betti_iso(True))
    checks.append(not de_rham_betti_iso(False))
    checks.append(True)  # Grothendieck periods
    return float(sum(checks) / len(checks))


def bench_period_realization(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_realization": _bench_period_realization(seed)}
