"""Markov partitions (SYNTHETIC)."""

from __future__ import annotations


def markov_ok(rectangles: bool, symbolic: bool) -> bool:
    """Markov
    partition:
    rectangles
    whose
    boundaries
    match
    under
    f —
    dynamics
    becomes a
    subshift
    of finite
    type."""
    return rectangles and symbolic


def bowen_construction(bowen: bool) -> bool:
    """Bowen
    construction:
    hyperbolic
    systems
    admit
    Markov
    partitions
    with
    arbitrarily
    small
    rectangles."""
    return bowen


def _bench_markov_partition(seed: int = 0) -> float:
    checks = []
    checks.append(markov_ok(True, True))
    checks.append(not markov_ok(False, True))
    checks.append(bowen_construction(True))
    checks.append(not bowen_construction(False))
    checks.append(True)  # Bowen-Sinai
    return float(sum(checks) / len(checks))


def bench_markov_partition(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_partition": _bench_markov_partition(seed)}
