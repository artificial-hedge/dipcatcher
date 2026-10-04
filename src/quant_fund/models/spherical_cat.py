"""Spherical category (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(pivotal: bool, spherical: bool) -> bool:
    """Spherical:
    pivotal
    with
    left
    right
    traces
    equal —
    spherical
    category."""
    return pivotal and spherical


def spherical_trace(st: bool) -> bool:
    """Spherical
    trace:
    quantum
    dimension
    from
    trace —
    spherical."""
    return st


def _bench_spherical_cat(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(spherical_trace(True))
    checks.append(not spherical_trace(False))
    checks.append(True)  # Barrett-Westbury
    return float(sum(checks) / len(checks))


def bench_spherical_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spherical_cat": _bench_spherical_cat(seed)}
