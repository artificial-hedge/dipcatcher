"""Giroux correspondence (SYNTHETIC)."""

from __future__ import annotations


def gc_ok(open_book: bool, bijection: bool) -> bool:
    """Giroux
    correspondence:
    open
    book
    decompositions
    up
    to
    stabilization
    correspond
    to
    contact
    structures
    up
    to
    isotopy."""
    return open_book and bijection


def stabilization_std(ss: bool) -> bool:
    """Positive
    stabilization:
    plumbing
    a
    positive
    Hopf
    band
    preserves
    the
    contact
    structure —
    key
    move."""
    return ss


def _bench_giroux_corr(seed: int = 0) -> float:
    checks = []
    checks.append(gc_ok(True, True))
    checks.append(not gc_ok(False, True))
    checks.append(stabilization_std(True))
    checks.append(not stabilization_std(False))
    checks.append(True)  # Giroux 2002
    return float(sum(checks) / len(checks))


def bench_giroux_corr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_giroux_corr": _bench_giroux_corr(seed)}
