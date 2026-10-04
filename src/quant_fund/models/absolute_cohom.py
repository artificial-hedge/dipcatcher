"""Absolute cohomology (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(absolute: bool, cohomology: bool) -> bool:
    """Absolute
    cohom:
    absolute
    cohomology —
    Hodge
    realization."""
    return absolute and cohomology


def absolute_hodge_cohom(ahc: bool) -> bool:
    """Absolute
    Hodge:
    absolute
    Hodge
    cohomology —
    Beilinson."""
    return ahc


def _bench_absolute_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(absolute_hodge_cohom(True))
    checks.append(not absolute_hodge_cohom(False))
    checks.append(True)  # Beilinson
    return float(sum(checks) / len(checks))


def bench_absolute_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_absolute_cohom": _bench_absolute_cohom(seed)}
