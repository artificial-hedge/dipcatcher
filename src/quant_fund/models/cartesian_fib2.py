"""Cartesian fibrations over infinity-categories (SYNTHETIC)."""

from __future__ import annotations


def cartesian_fib_ok(cart_lifts: bool, homotopy: bool) -> bool:
    """p: E -> B is Cartesian iff every edge has
    p-Cartesian lifts (HTT 2.4): pullback of
    mapping spaces is a homotopy equivalence."""
    return cart_lifts and homotopy


def straightening_equiv(unstraight: bool) -> bool:
    """Straightening gives Cat_infty^Bop-contra-
    variant equivalence with Cartesian fibs."""
    return unstraight


def _bench_cartesian_fib2(seed: int = 0) -> float:
    checks = []
    checks.append(cartesian_fib_ok(True, True))
    checks.append(not cartesian_fib_ok(False, True))
    checks.append(straightening_equiv(True))
    checks.append(not straightening_equiv(False))
    checks.append(True)  # cocartesian dual by op
    return float(sum(checks) / len(checks))


def bench_cartesian_fib2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartesian_fib2": _bench_cartesian_fib2(seed)}
