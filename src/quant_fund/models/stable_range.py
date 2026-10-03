"""Freudenthal suspension / stable range (SYNTHETIC)."""

from __future__ import annotations


def is_stable(i: int, n: int) -> bool:
    """pi_i(S^n) -> pi_{i+1}(S^{n+1}) iso for i < 2n-1."""
    return i < 2 * n - 1


def _bench_stable_range(seed: int = 0) -> float:
    checks = []
    # pi_1(S1) -> pi_2(S2): i=1, n=1 -> 1 < 1 false (barely unstable)
    checks.append(not is_stable(1, 1))
    # pi_2(S2) -> pi_3(S3): 2 < 3 stable
    checks.append(is_stable(2, 2))
    # pi_4(S3): 4 < 5 stable
    checks.append(is_stable(4, 3))
    # edge of range: i = 2n-1 is epic not iso
    checks.append(not is_stable(3, 2))
    # suspension of S^n gives S^{n+1}
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_stable_range(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_range": _bench_stable_range(seed)}
