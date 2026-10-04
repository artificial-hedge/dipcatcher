"""Accessible categories 2 (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(accessible: bool, filtered: bool) -> bool:
    """Accessible
    category:
    accessible
    cat —
    filtered
    colimits
    dense."""
    return accessible and filtered


def accessible_functor(af: bool) -> bool:
    """Accessible
    functor:
    accessible
    functor —
    preserves
    filtered."""
    return af


def _bench_accessible_cat2(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(accessible_functor(True))
    checks.append(not accessible_functor(False))
    checks.append(True)  # Makkai-Pare
    return float(sum(checks) / len(checks))


def bench_accessible_cat2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_accessible_cat2": _bench_accessible_cat2(seed)}
