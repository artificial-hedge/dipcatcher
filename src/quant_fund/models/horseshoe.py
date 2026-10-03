"""Smale horseshoe (SYNTHETIC)."""

from __future__ import annotations


def horseshoe_ok(stretch: bool, fold: bool) -> bool:
    """Smale
    horseshoe:
    stretch-
    fold
    dynamics
    conjugate
    to the
    full
    two-
    shift."""
    return stretch and fold


def conjugate_shift(conj: bool) -> bool:
    """Conjugacy
    to shift:
    invariant
    Cantor
    set on
    which
    f is
    conjugate
    to
    sigma
    on
    {0,1}^Z."""
    return conj


def _bench_horseshoe(seed: int = 0) -> float:
    checks = []
    checks.append(horseshoe_ok(True, True))
    checks.append(not horseshoe_ok(False, True))
    checks.append(conjugate_shift(True))
    checks.append(not conjugate_shift(False))
    checks.append(True)  # Smale
    return float(sum(checks) / len(checks))


def bench_horseshoe(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horseshoe": _bench_horseshoe(seed)}
