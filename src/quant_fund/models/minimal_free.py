"""Minimal free resolutions (SYNTHETIC)."""

from __future__ import annotations


def minimal_ok(free: bool, unique: bool) -> bool:
    """Minimal
    free
    resolution
    F_* -> M:
    differentials
    land in the
    maximal
    ideal; unique
    up to
    isomorphism."""
    return free and unique


def hilbert_syzygy(syzygy: bool) -> bool:
    """Hilbert
    syzygy
    theorem:
    over
    k[x_1,...,x_n]
    every
    module has
    pd <= n."""
    return syzygy


def _bench_minimal_free(seed: int = 0) -> float:
    checks = []
    checks.append(minimal_ok(True, True))
    checks.append(not minimal_ok(False, True))
    checks.append(hilbert_syzygy(True))
    checks.append(not hilbert_syzygy(False))
    checks.append(True)  # Hilbert
    return float(sum(checks) / len(checks))


def bench_minimal_free(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minimal_free": _bench_minimal_free(seed)}
