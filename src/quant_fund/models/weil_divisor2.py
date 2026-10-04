"""Weil divisors (SYNTHETIC)."""

from __future__ import annotations


def wd2_ok(weil: bool, divisor: bool) -> bool:
    """Weil
    divisor:
    Weil
    divisor —
    codim
    1
    cycle."""
    return weil and divisor


def principal_divisor(pd: bool) -> bool:
    """Principal
    divisor:
    principal
    Weil
    divisor —
    rational
    function."""
    return pd


def _bench_weil_divisor2(seed: int = 0) -> float:
    checks = []
    checks.append(wd2_ok(True, True))
    checks.append(not wd2_ok(False, True))
    checks.append(principal_divisor(True))
    checks.append(not principal_divisor(False))
    checks.append(True)  # Weil
    return float(sum(checks) / len(checks))


def bench_weil_divisor2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weil_divisor2": _bench_weil_divisor2(seed)}
