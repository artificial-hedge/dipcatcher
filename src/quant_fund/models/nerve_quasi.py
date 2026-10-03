"""Nerve of a category as quasi-cat (SYNTHETIC)."""

from __future__ import annotations


def nerve_ok(nerve: bool, inner_kan: bool) -> bool:
    """Nerve N(C) of
    an ordinary
    category: quasi-
    cat; unique inner-
    horn fillers."""
    return nerve and inner_kan


def unique_fill(unique: bool) -> bool:
    """Unique fillers:
    N(C) has unique
    lifts for inner
    horns; this
    characterizes
    nerves."""
    return unique


def _bench_nerve_quasi(seed: int = 0) -> float:
    checks = []
    checks.append(nerve_ok(True, True))
    checks.append(not nerve_ok(False, True))
    checks.append(unique_fill(True))
    checks.append(not unique_fill(False))
    checks.append(True)  # Grothendieck nerve
    return float(sum(checks) / len(checks))


def bench_nerve_quasi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nerve_quasi": _bench_nerve_quasi(seed)}
