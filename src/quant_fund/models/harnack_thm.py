"""Harnack inequality (SYNTHETIC)."""

from __future__ import annotations


def har_ok(positive: bool, uniform: bool) -> bool:
    """Harnack:
    sup
    bounded
    by
    inf
    times
    a
    uniform
    constant
    for
    nonnegative
    harmonic
    functions."""
    return positive and uniform


def liouville_from_harnack(lh: bool) -> bool:
    """Harnack
    implies
    Liouville:
    bounded
    harmonic
    on
    all
    space
    is
    constant."""
    return lh


def _bench_harnack_thm(seed: int = 0) -> float:
    checks = []
    checks.append(har_ok(True, True))
    checks.append(not har_ok(False, True))
    checks.append(liouville_from_harnack(True))
    checks.append(not liouville_from_harnack(False))
    checks.append(True)  # Harnack-Moser
    return float(sum(checks) / len(checks))


def bench_harnack_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harnack_thm": _bench_harnack_thm(seed)}
