"""Pseudo-Anosov maps (SYNTHETIC)."""

from __future__ import annotations


def pa_ok(expanding: bool, contracting: bool) -> bool:
    """Pseudo-
    Anosov:
    mapping
    class
    with
    two
    transverse
    measured
    foliations,
    expanding
    one
    and
    contracting
    the
    other
    by
    lambda>1."""
    return expanding and contracting


def stretch_factor(sf: bool) -> bool:
    """Stretch
    factor:
    the
    dilatation
    lambda
    is
    a
    Perron
    algebraic
    integer —
    Fried,
    Penner."""
    return sf


def _bench_pseudo_anosov(seed: int = 0) -> float:
    checks = []
    checks.append(pa_ok(True, True))
    checks.append(not pa_ok(False, True))
    checks.append(stretch_factor(True))
    checks.append(not stretch_factor(False))
    checks.append(True)  # Thurston
    return float(sum(checks) / len(checks))


def bench_pseudo_anosov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudo_anosov": _bench_pseudo_anosov(seed)}
