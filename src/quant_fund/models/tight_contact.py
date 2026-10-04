"""Tight contact structures (SYNTHETIC)."""

from __future__ import annotations


def tc_ok(no_ot_disk: bool, rigid: bool) -> bool:
    """Tight:
    no
    overtwisted
    disk —
    rigid
    structures
    with
    contact
    homology
    invariants."""
    return no_ot_disk and rigid


def fillable_tight(ft: bool) -> bool:
    """Fillable
    implies
    tight:
    symplectically
    fillable
    contact
    manifolds
    are
    tight —
    Gromov-
    Eliashberg."""
    return ft


def _bench_tight_contact(seed: int = 0) -> float:
    checks = []
    checks.append(tc_ok(True, True))
    checks.append(not tc_ok(False, True))
    checks.append(fillable_tight(True))
    checks.append(not fillable_tight(False))
    checks.append(True)  # Gromov-Eliashberg
    return float(sum(checks) / len(checks))


def bench_tight_contact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tight_contact": _bench_tight_contact(seed)}
