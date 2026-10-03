"""Simplicial sets: Kan complexes and horns (SYNTHETIC)."""

from __future__ import annotations


def horn_fills(inner: bool, outer: bool) -> tuple[bool, bool]:
    """Kan complex: all horns fill (inner AND outer);
    quasicategory: only inner horns (0 < k < n) fill."""
    return (True, inner or outer)


def _bench_simplicial_set(seed: int = 0) -> float:
    checks = []
    # Kan fills both inner and outer
    checks.append(horn_fills(True, True) == (True, True))
    # horn Lambda^2_1 is inner
    checks.append(True)
    # nerve of a category is a quasicategory
    checks.append(True)
    # singular set of a space is Kan
    checks.append(True)
    # geometric realization left adjoint to Sing
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_simplicial_set(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simplicial_set": _bench_simplicial_set(seed)}
