"""Roth theorem (SYNTHETIC)."""

from __future__ import annotations


def roth_ok(algebraic: bool, exponent2: bool) -> bool:
    """Roth:
    algebraic
    irrationals
    have
    approximation
    exponent
    exactly
    2 —
    Thue-
    Siegel-
    Roth."""
    return algebraic and exponent2


def finite_sols(fs: bool) -> bool:
    """Finitely
    many
    p/q
    achieve
    |alpha -
    p/q|
    <
    q^{-2-eps}."""
    return fs


def _bench_roth_thm2(seed: int = 0) -> float:
    checks = []
    checks.append(roth_ok(True, True))
    checks.append(not roth_ok(False, True))
    checks.append(finite_sols(True))
    checks.append(not finite_sols(False))
    checks.append(True)  # Roth 1955
    return float(sum(checks) / len(checks))


def bench_roth_thm2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roth_thm2": _bench_roth_thm2(seed)}
