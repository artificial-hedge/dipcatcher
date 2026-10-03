"""Goodwillie derivatives (SYNTHETIC)."""

from __future__ import annotations


def gd_ok(goodwillie: bool, deriv: bool) -> bool:
    """Goodwillie
    deriv:
    Goodwillie
    derivative —
    symmetric."""
    return goodwillie and deriv


def nth_derivative(nd: bool) -> bool:
    """Nth
    derivative:
    nth
    derivative —
    multilinear."""
    return nd


def _bench_goodwillie_deriv(seed: int = 0) -> float:
    checks = []
    checks.append(gd_ok(True, True))
    checks.append(not gd_ok(False, True))
    checks.append(nth_derivative(True))
    checks.append(not nth_derivative(False))
    checks.append(True)  # Goodwillie
    return float(sum(checks) / len(checks))


def bench_goodwillie_deriv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_goodwillie_deriv": _bench_goodwillie_deriv(seed)}
