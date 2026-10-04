"""p-adic local field: valuation and ultrametric (SYNTHETIC)."""

from __future__ import annotations


def v_p(x: int, p: int) -> float:
    """p-adic valuation of nonzero integer."""
    if x == 0:
        return float("inf")
    n = 0
    x = abs(x)
    while x % p == 0:
        x //= p
        n += 1
    return float(n)


def norm_p(x: int, p: int) -> float:
    """|x|_p = p^{-v_p(x)}."""
    return float(p ** (-v_p(x, p)))


def _bench_local_field(seed: int = 0) -> float:
    checks = []
    checks.append(v_p(40, 5) == 1.0)
    checks.append(v_p(8, 2) == 3.0)
    checks.append(norm_p(8, 2) == 1 / 8)
    # ultrametric: |x+y|_p <= max(|x|_p, |y|_p) -- try several pairs
    ok = all(
        norm_p(x + y, 3) <= max(norm_p(x, 3), norm_p(y, 3)) + 1e-12
        for x in range(1, 15)
        for y in range(1, 15)
        if x + y != 0
    )
    checks.append(ok)
    # multiplicativity |xy|_p = |x|_p |y|_p
    checks.append(abs(norm_p(6, 3) - norm_p(2, 3) * norm_p(3, 3)) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_local_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_field": _bench_local_field(seed)}
