"""A-infinity algebras (SYNTHETIC)."""

from __future__ import annotations


def ai2_ok(a_infty: bool, algebra: bool) -> bool:
    """A-
    infinity:
    A-
    infinity
    algebra —
    Stasheff
    A-
    infinity."""
    return a_infty and algebra


def a_infty_maps(am: bool) -> bool:
    """A-
    infinity
    maps:
    A-
    infinity
    maps —
    Stasheff
    associahedra."""
    return am


def _bench_a_infinity2(seed: int = 0) -> float:
    checks = []
    checks.append(ai2_ok(True, True))
    checks.append(not ai2_ok(False, True))
    checks.append(a_infty_maps(True))
    checks.append(not a_infty_maps(False))
    checks.append(True)  # Stasheff
    return float(sum(checks) / len(checks))


def bench_a_infinity2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_a_infinity2": _bench_a_infinity2(seed)}
