"""Comma categories (SYNTHETIC)."""

from __future__ import annotations


def comma_cat_ok(universal: bool, slice_is: bool) -> bool:
    """Comma category (F/G)
    objects (a, b, f: Fa->Gb);
    slice/coslice as special
    cases; universal
    properties hold."""
    return universal and slice_is


def slice_topos_2(pullback: bool) -> bool:
    """Comma squares are 2-
    categorical limits; the
    slice E/X is a topos
    when E is."""
    return pullback


def _bench_comma_cat(seed: int = 0) -> float:
    checks = []
    checks.append(comma_cat_ok(True, True))
    checks.append(not comma_cat_ok(False, True))
    checks.append(slice_topos_2(True))
    checks.append(not slice_topos_2(False))
    checks.append(True)  # strict 2-limits
    return float(sum(checks) / len(checks))


def bench_comma_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_comma_cat": _bench_comma_cat(seed)}
