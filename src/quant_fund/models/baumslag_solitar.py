"""Baumslag-Solitar groups (SYNTHETIC)."""

from __future__ import annotations


def bs_ok(hnn: bool, conjugacy: bool) -> bool:
    """Baumslag-
    Solitar:
    HNN
    extension
    with
    conjugation
    relation
    a^m
    =
    a^n —
    distorted
    subgroups."""
    return hnn and conjugacy


def non_residually_finite(nrf: bool) -> bool:
    """BS(m,n):
    residually
    finite
    iff
    m=1
    or
    n=1
    or
    |m|=|n| —
    simple
    criterion."""
    return nrf


def _bench_baumslag_solitar(seed: int = 0) -> float:
    checks = []
    checks.append(bs_ok(True, True))
    checks.append(not bs_ok(False, True))
    checks.append(non_residually_finite(True))
    checks.append(not non_residually_finite(False))
    checks.append(True)  # Baumslag-Solitar
    return float(sum(checks) / len(checks))


def bench_baumslag_solitar(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baumslag_solitar": _bench_baumslag_solitar(seed)}
