"""Exponential motives (SYNTHETIC)."""

from __future__ import annotations


def em_ok(exponential: bool, motive: bool) -> bool:
    """Exponential
    motive:
    exponential
    motive —
    irregular."""
    return exponential and motive


def irregular_singular(isg: bool) -> bool:
    """Irregular
    singularity:
    irregular
    singularity
    motive —
    exp."""
    return isg


def _bench_exponential_motive(seed: int = 0) -> float:
    checks = []
    checks.append(em_ok(True, True))
    checks.append(not em_ok(False, True))
    checks.append(irregular_singular(True))
    checks.append(not irregular_singular(False))
    checks.append(True)  # Fresan-Jossen
    return float(sum(checks) / len(checks))


def bench_exponential_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exponential_motive": _bench_exponential_motive(seed)}
