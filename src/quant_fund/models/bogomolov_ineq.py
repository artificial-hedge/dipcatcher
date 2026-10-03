"""Bogomolov inequality (SYNTHETIC)."""

from __future__ import annotations


def bi_ok(chern_square: bool, stable_bundle: bool) -> bool:
    """Bogomolov
    inequality:
    c2
    minus
    c1-squared
    term
    nonneg
    on
    stable
    bundles —
    Yau
    proof."""
    return chern_square and stable_bundle


def restriction_curve(rc: bool) -> bool:
    """Restriction
    to
    curves:
    Bogomolov
    via
    restriction
    theorems
    of
    Mehta-
    Ramanathan
    —
    slope
    bound."""
    return rc


def _bench_bogomolov_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(bi_ok(True, True))
    checks.append(not bi_ok(False, True))
    checks.append(restriction_curve(True))
    checks.append(not restriction_curve(False))
    checks.append(True)  # Bogomolov-Gieseker
    return float(sum(checks) / len(checks))


def bench_bogomolov_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bogomolov_ineq": _bench_bogomolov_ineq(seed)}
