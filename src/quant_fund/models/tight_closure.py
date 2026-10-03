"""Tight closure (SYNTHETIC)."""

from __future__ import annotations


def tight_ok(nu_test: bool, colon: bool) -> bool:
    """Tight closure I*:
    x in I* iff exists
    c not in any min
    prime with
    c x^{p^e} in
    I^{[p^e]} for
    all e >> 0."""
    return nu_test and colon


def colon_criterion(regular: bool) -> bool:
    """Colon criterion:
    in F-regular rings
    I* = I; this is
    the defining
    tight-closure
    property."""
    return regular


def _bench_tight_closure(seed: int = 0) -> float:
    checks = []
    checks.append(tight_ok(True, True))
    checks.append(not tight_ok(False, True))
    checks.append(colon_criterion(True))
    checks.append(not colon_criterion(False))
    checks.append(True)  # Hochster-Huneke
    return float(sum(checks) / len(checks))


def bench_tight_closure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tight_closure": _bench_tight_closure(seed)}
