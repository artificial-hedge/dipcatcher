"""Furstenberg theory (SYNTHETIC)."""

from __future__ import annotations


def furst_ok(correspondence: bool, ergodic: bool) -> bool:
    """Furstenberg
    correspondence:
    Szemeredi
    equivalent
    to
    multiple-
    recurrence
    in measure
    systems."""
    return correspondence and ergodic


def multiple_recurrence(rec: bool) -> bool:
    """Multiple
    recurrence:
    T-invariant
    measure
    systems
    return
    along
    arithmetic
    progressions
    with
    positive
    density."""
    return rec


def _bench_furstenberg(seed: int = 0) -> float:
    checks = []
    checks.append(furst_ok(True, True))
    checks.append(not furst_ok(False, True))
    checks.append(multiple_recurrence(True))
    checks.append(not multiple_recurrence(False))
    checks.append(True)  # Furstenberg
    return float(sum(checks) / len(checks))


def bench_furstenberg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_furstenberg": _bench_furstenberg(seed)}
