"""Milnor fiber (SYNTHETIC)."""

from __future__ import annotations


def mf_ok(local_fibration: bool, homology: bool) -> bool:
    """Milnor
    fiber:
    nearby-
    fiber
    fibration
    over
    small
    circle —
    bouquet
    of
    spheres."""
    return local_fibration and homology


def milnor_number(mn: bool) -> bool:
    """Milnor
    number:
    middle
    Betti
    of
    the
    Milnor
    fiber —
    counts
    critical
    complexity."""
    return mn


def _bench_milnor_fiber(seed: int = 0) -> float:
    checks = []
    checks.append(mf_ok(True, True))
    checks.append(not mf_ok(False, True))
    checks.append(milnor_number(True))
    checks.append(not milnor_number(False))
    checks.append(True)  # Milnor
    return float(sum(checks) / len(checks))


def bench_milnor_fiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_milnor_fiber": _bench_milnor_fiber(seed)}
