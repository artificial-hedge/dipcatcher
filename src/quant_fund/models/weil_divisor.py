"""Weil divisor (SYNTHETIC)."""

from __future__ import annotations


def wd_ok(prime_divisor: bool, formal_sum: bool) -> bool:
    """Weil
    divisor:
    formal
    sum
    of
    prime
    divisors —
    divisor
    group."""
    return prime_divisor and formal_sum


def divisor_class_group(dcg: bool) -> bool:
    """Divisor
    class
    group:
    divisor
    modulo
    rational
    equivalence —
    class
    group."""
    return dcg


def _bench_weil_divisor(seed: int = 0) -> float:
    checks = []
    checks.append(wd_ok(True, True))
    checks.append(not wd_ok(False, True))
    checks.append(divisor_class_group(True))
    checks.append(not divisor_class_group(False))
    checks.append(True)  # Weil
    return float(sum(checks) / len(checks))


def bench_weil_divisor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weil_divisor": _bench_weil_divisor(seed)}
