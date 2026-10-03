"""Induction schema soundness on finite checks (SYNTHETIC)."""

from __future__ import annotations

from collections.abc import Callable


def induction_check(base: bool, step: Callable[[int], bool], n: int) -> bool:
    """P(0) and forall k<n P(k) -> P(k+1) implies P(n)."""
    if not base:
        return False
    return all(step(k) for k in range(n + 1))


def _bench_finitary_induct(seed: int = 0) -> float:
    checks = []

    # P(n): sum 0..n = n(n+1)/2 verified by induction step data
    def gauss(k: int) -> bool:
        return sum(range(k + 1)) == k * (k + 1) // 2

    checks.append(induction_check(True, gauss, 20))

    # P(n): 2^n > n for all n
    def pow_bound(k: int) -> bool:
        return (1 << k) > k

    checks.append(induction_check(True, pow_bound, 10))
    # false base fails
    checks.append(not induction_check(False, gauss, 5))
    # P(n): n < n+1 trivially
    checks.append(induction_check(True, lambda k: k < k + 1, 50))
    return float(sum(checks) / len(checks))


def bench_finitary_induct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_finitary_induct": _bench_finitary_induct(seed)}
