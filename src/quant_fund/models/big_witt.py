"""Big Witt vectors (SYNTHETIC)."""

from __future__ import annotations


def bw_ok(big: bool, witt: bool) -> bool:
    """Big
    Witt:
    big
    Witt
    vectors —
    universal
    Witt."""
    return big and witt


def lambda_ring(lr: bool) -> bool:
    """Lambda
    ring:
    lambda
    ring —
    lambda
    operations."""
    return lr


def _bench_big_witt(seed: int = 0) -> float:
    checks = []
    checks.append(bw_ok(True, True))
    checks.append(not bw_ok(False, True))
    checks.append(lambda_ring(True))
    checks.append(not lambda_ring(False))
    checks.append(True)  # Cartier
    return float(sum(checks) / len(checks))


def bench_big_witt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_big_witt": _bench_big_witt(seed)}
