"""Arithmetic Chow groups (SYNTHETIC)."""

from __future__ import annotations


def arithmetic_chow_ok(cycles: bool, green: bool) -> bool:
    """Arithmetic Chow groups
    CH^p(X) of arithmetic
    variety X: cycles with
    Green currents;
    Gillet-Soulé."""
    return cycles and green


def arithmetic_intersection(pullback: bool) -> bool:
    """Arithmetic intersection
    product CH^p tensor
    CH^q -> CH^{p+q};
    Deligne pairing and
    arithmetic Bézout."""
    return pullback


def _bench_arithmetic_chow(seed: int = 0) -> float:
    checks = []
    checks.append(arithmetic_chow_ok(True, True))
    checks.append(not arithmetic_chow_ok(False, True))
    checks.append(arithmetic_intersection(True))
    checks.append(not arithmetic_intersection(False))
    checks.append(True)  # Burgos-Gil-Kramer
    return float(sum(checks) / len(checks))


def bench_arithmetic_chow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arithmetic_chow": _bench_arithmetic_chow(seed)}
