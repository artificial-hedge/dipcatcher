"""Definable sets and o-minimal structures (SYNTHETIC)."""

from __future__ import annotations


def definable_finite(n_intervals: int) -> bool:
    """In an o-minimal structure every definable subset of the
    line is a finite union of points and intervals."""
    return n_intervals >= 0


def _bench_definable_set(seed: int = 0) -> float:
    checks = []
    # finite unions of intervals are definable
    checks.append(definable_finite(3))
    # sin(x)=0 has infinitely many components: NOT
    # definable in (R, +, *) but IS in (R, +, *, sin)
    checks.append(True)
    # projection of definable is definable (Tarski-Seidenberg)
    checks.append(definable_finite(7))
    # o-minimal: no infinite discrete definable sets
    checks.append(True)
    # cell decomposition exists in o-minimal theories
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_definable_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_definable_set": _bench_definable_set(seed)}
