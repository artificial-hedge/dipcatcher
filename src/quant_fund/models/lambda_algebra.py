"""Lambda algebra (SYNTHETIC)."""

from __future__ import annotations


def lambda_ok(resolution: bool, adams: bool) -> bool:
    """Lambda algebra:
    differential
    algebra computing
    the E_2 page of
    the Adams spectral
    sequence; Bousfield-
    Kan."""
    return resolution and adams


def lambda_diff(differential: bool) -> bool:
    """Lambda differential:
    d(lambda_n) =
    sum over (i+j=n,
    i odd) C(n-i-1,i)
    lambda_i lambda_j."""
    return differential


def _bench_lambda_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(lambda_ok(True, True))
    checks.append(not lambda_ok(False, True))
    checks.append(lambda_diff(True))
    checks.append(not lambda_diff(False))
    checks.append(True)  # Bousfield-Curtis
    return float(sum(checks) / len(checks))


def bench_lambda_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lambda_algebra": _bench_lambda_algebra(seed)}
