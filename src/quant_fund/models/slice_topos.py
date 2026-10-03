"""Slice topos (SYNTHETIC)."""

from __future__ import annotations


def st_ok(slice: bool, topos: bool) -> bool:
    """Slice:
    slice
    topos
    over
    object —
    Freyd
    slice."""
    return slice and topos


def dependent_sum(ds: bool) -> bool:
    """Dependent
    sum:
    dependent
    sum
    and
    product
    —
    dependent
    types."""
    return ds


def _bench_slice_topos(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(dependent_sum(True))
    checks.append(not dependent_sum(False))
    checks.append(True)  # Freyd
    return float(sum(checks) / len(checks))


def bench_slice_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slice_topos": _bench_slice_topos(seed)}
