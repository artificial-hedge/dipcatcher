"""Frobenius group structure S3 = C3 . C2 (SYNTHETIC)."""

from __future__ import annotations


def frobenius_kernel_size(n: int) -> int:
    """S3: kernel = A3 = C3 has order 3."""
    return n // 2


def irreducible_degrees() -> list[int]:
    """S3 has irreps of degree 1,1,2: two from complement, one induced
    from kernel non-trivial chars."""
    return [1, 1, 2]


def _bench_frobenius_group(seed: int = 0) -> float:
    checks = []
    checks.append(frobenius_kernel_size(6) == 3)
    degs = irreducible_degrees()
    # sum of squares = |G|
    checks.append(sum(d * d for d in degs) == 6)
    # number of irreps = number of conjugacy classes = 3
    checks.append(len(degs) == 3)
    # two 1-d reps factor through abelianization C2
    checks.append(degs.count(1) == 2)
    # induced reps from nontrivial C3 chars give the 2-d
    checks.append(2 in degs)
    return float(sum(checks) / len(checks))


def bench_frobenius_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frobenius_group": _bench_frobenius_group(seed)}
