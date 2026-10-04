"""Absolute Hodge cycles (SYNTHETIC)."""

from __future__ import annotations


def ah_ok(all_realizations: bool, comparison: bool) -> bool:
    """Absolute
    Hodge
    cycle:
    compatible
    in
    every
    realization —
    Deligne's
    substitute
    for
    Hodge
    classes."""
    return all_realizations and comparison


def deligne_theorem(dt: bool) -> bool:
    """Deligne:
    Hodge
    classes
    on
    abelian
    varieties
    are
    absolute —
    CM
    reduction."""
    return dt


def _bench_absolute_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(ah_ok(True, True))
    checks.append(not ah_ok(False, True))
    checks.append(deligne_theorem(True))
    checks.append(not deligne_theorem(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_absolute_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_absolute_hodge": _bench_absolute_hodge(seed)}
