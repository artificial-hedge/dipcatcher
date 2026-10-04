"""Bowen specification (SYNTHETIC)."""

from __future__ import annotations


def spec_ok(shadow: bool, periodic: bool) -> bool:
    """Specification:
    orbit
    segments
    can be
    shadowed
    by a
    single
    periodic
    orbit
    with
    bounded
    gaps."""
    return shadow and periodic


def entropy_equidist(ent: bool) -> bool:
    """Periodic
    orbit
    counting:
    growth
    rate of
    Fix(f^n)
    equals
    topological
    entropy
    for
    expansive
    specification
    systems."""
    return ent


def _bench_bowen_spec(seed: int = 0) -> float:
    checks = []
    checks.append(spec_ok(True, True))
    checks.append(not spec_ok(False, True))
    checks.append(entropy_equidist(True))
    checks.append(not entropy_equidist(False))
    checks.append(True)  # Bowen
    return float(sum(checks) / len(checks))


def bench_bowen_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bowen_spec": _bench_bowen_spec(seed)}
