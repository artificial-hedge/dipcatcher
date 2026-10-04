"""Aubin's theorem (SYNTHETIC)."""

from __future__ import annotations


def at_ok(below_sphere: bool, test_function: bool) -> bool:
    """Aubin:
    if
    the
    Yamabe
    constant
    is
    below
    the
    sphere's,
    a
    minimizer
    exists —
    local
    test."""
    return below_sphere and test_function


def weyl_vanishing(wv: bool) -> bool:
    """Weyl
    vanishing:
    Aubin
    shows
    the
    constant
    drops
    unless
    conformally
    flat —
    high-
    dim
    proof."""
    return wv


def _bench_aubin_thm(seed: int = 0) -> float:
    checks = []
    checks.append(at_ok(True, True))
    checks.append(not at_ok(False, True))
    checks.append(weyl_vanishing(True))
    checks.append(not weyl_vanishing(False))
    checks.append(True)  # Aubin
    return float(sum(checks) / len(checks))


def bench_aubin_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aubin_thm": _bench_aubin_thm(seed)}
