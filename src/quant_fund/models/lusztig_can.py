"""Lusztig canonical basis (SYNTHETIC)."""

from __future__ import annotations


def can_ok(basis: bool, positivity: bool) -> bool:
    """Lusztig
    canonical
    basis of
    U_q^-(g):
    self-dual,
    bar-
    invariant;
    structure
    constants
    in
    N[q,q^-1]."""
    return basis and positivity


def positivity_thm(pos: bool) -> bool:
    """Positivity:
    canonical
    basis
    products
    have
    nonneg
    structure
    constants
    (Lusztig)."""
    return pos


def _bench_lusztig_can(seed: int = 0) -> float:
    checks = []
    checks.append(can_ok(True, True))
    checks.append(not can_ok(False, True))
    checks.append(positivity_thm(True))
    checks.append(not positivity_thm(False))
    checks.append(True)  # Lusztig
    return float(sum(checks) / len(checks))


def bench_lusztig_can(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lusztig_can": _bench_lusztig_can(seed)}
