"""Primitive-recursive definitions of arithmetic (SYNTHETIC)."""

from __future__ import annotations


def pr_add(a: int, b: int) -> int:
    """add via successor recursion on b."""
    if b == 0:
        return a
    return pr_add(a + 1, b - 1)


def pr_mul(a: int, b: int) -> int:
    """mul via repeated add."""
    if b == 0:
        return 0
    return pr_add(a, pr_mul(a, b - 1))


def pr_fact(n: int) -> int:
    """factorial via primitive recursion."""
    if n == 0:
        return 1
    return pr_mul(n, pr_fact(n - 1))


def pr_pred(n: int) -> int:
    """cutoff predecessor."""
    return max(0, n - 1)


def _bench_primitive_recursion(seed: int = 0) -> float:
    checks = []
    checks.append(pr_add(3, 4) == 7)
    checks.append(pr_mul(3, 4) == 12)
    checks.append(pr_fact(5) == 120)
    checks.append(pr_fact(0) == 1)
    checks.append(pr_pred(0) == 0)
    checks.append(pr_pred(9) == 8)
    return float(sum(checks) / len(checks))


def bench_primitive_recursion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primitive_recursion": _bench_primitive_recursion(seed)}
