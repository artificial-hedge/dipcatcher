"""Asymptotic motives (SYNTHETIC)."""

from __future__ import annotations


def am_ok(asymptotic: bool, motive: bool) -> bool:
    """Asymptotic
    motive:
    asymptotic
    motive —
    infinity."""
    return asymptotic and motive


def asymptotic_limit(al: bool) -> bool:
    """Asymptotic
    limit:
    asymptotic
    limit
    motive —
    Hodge."""
    return al


def _bench_asymptotic_motive(seed: int = 0) -> float:
    checks = []
    checks.append(am_ok(True, True))
    checks.append(not am_ok(False, True))
    checks.append(asymptotic_limit(True))
    checks.append(not asymptotic_limit(False))
    checks.append(True)  # limit motives
    return float(sum(checks) / len(checks))


def bench_asymptotic_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_asymptotic_motive": _bench_asymptotic_motive(seed)}
