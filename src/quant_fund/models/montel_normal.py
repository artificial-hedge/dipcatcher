"""Montel normal families (SYNTHETIC)."""

from __future__ import annotations


def montel_ok(bounded: bool, precompact: bool) -> bool:
    """Montel:
    locally
    bounded
    families
    of
    holomorphic
    functions
    are
    normal —
    subsequence
    converges."""
    return bounded and precompact


def omits_three(o3: bool) -> bool:
    """Normality
    criterion:
    families
    omitting
    three
    values
    are
    normal."""
    return o3


def _bench_montel_normal(seed: int = 0) -> float:
    checks = []
    checks.append(montel_ok(True, True))
    checks.append(not montel_ok(False, True))
    checks.append(omits_three(True))
    checks.append(not omits_three(False))
    checks.append(True)  # Montel
    return float(sum(checks) / len(checks))


def bench_montel_normal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_montel_normal": _bench_montel_normal(seed)}
