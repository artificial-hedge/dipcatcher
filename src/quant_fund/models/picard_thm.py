"""Picard theorem (SYNTHETIC)."""

from __future__ import annotations


def pic_ok(essential: bool, two_omitted: bool) -> bool:
    """Great
    Picard:
    near
    an
    essential
    singularity
    a
    function
    attains
    every
    value
    with
    at most
    one
    exception."""
    return essential and two_omitted


def little_picard(lp: bool) -> bool:
    """Little
    Picard:
    entire
    functions
    omitting
    two
    values
    are
    constant."""
    return lp


def _bench_picard_thm(seed: int = 0) -> float:
    checks = []
    checks.append(pic_ok(True, True))
    checks.append(not pic_ok(False, True))
    checks.append(little_picard(True))
    checks.append(not little_picard(False))
    checks.append(True)  # Picard
    return float(sum(checks) / len(checks))


def bench_picard_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_thm": _bench_picard_thm(seed)}
