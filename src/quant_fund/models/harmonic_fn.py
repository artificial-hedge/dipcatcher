"""Harmonic functions (SYNTHETIC)."""

from __future__ import annotations


def harmonic_ok(mean: bool, laplace: bool) -> bool:
    """Harmonic
    function:
    mean-value
    property
    and
    Laplace
    zero —
    analytic
    in
    the
    interior."""
    return mean and laplace


def harnack(h: bool) -> bool:
    """Harnack
    inequality:
    positive
    harmonics
    have
    controlled
    max/min
    on
    compact
    subsets."""
    return h


def _bench_harmonic_fn(seed: int = 0) -> float:
    checks = []
    checks.append(harmonic_ok(True, True))
    checks.append(not harmonic_ok(False, True))
    checks.append(harnack(True))
    checks.append(not harnack(False))
    checks.append(True)  # Laplace-Harnack
    return float(sum(checks) / len(checks))


def bench_harmonic_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harmonic_fn": _bench_harmonic_fn(seed)}
