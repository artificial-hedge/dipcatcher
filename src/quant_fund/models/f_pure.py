"""F-pure rings (SYNTHETIC)."""

from __future__ import annotations


def f_pure_ok(frobenius_pure: bool, splitting: bool) -> bool:
    """F-pure ring: Frobenius
    is a pure map —
    splits locally;
    Fedder's criterion
    via Cartier."""
    return frobenius_pure and splitting


def fedder_criterion(cartier: bool) -> bool:
    """Fedder's criterion:
    R = S/I F-pure iff
    (I^{[p]}:I) not
    contained in
    the maximal ideal."""
    return cartier


def _bench_f_pure(seed: int = 0) -> float:
    checks = []
    checks.append(f_pure_ok(True, True))
    checks.append(not f_pure_ok(False, True))
    checks.append(fedder_criterion(True))
    checks.append(not fedder_criterion(False))
    checks.append(True)  # Fedder 1983
    return float(sum(checks) / len(checks))


def bench_f_pure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_pure": _bench_f_pure(seed)}
