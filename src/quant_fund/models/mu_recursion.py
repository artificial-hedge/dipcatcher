"""mu-operator: unbounded minimization (bounded here) (SYNTHETIC)."""

from __future__ import annotations


def mu(f, lo: int = 0, hi: int = 10_000) -> int | None:
    """Least n in [lo, hi) with f(n) == 0, or None."""
    for n in range(lo, hi):
        if f(n) == 0:
            return n
    return None


def _bench_mu_recursion(seed: int = 0) -> float:
    checks = []
    # mu of n - 3 is 3
    checks.append(mu(lambda n: n - 3) == 3)
    # mu over a window starting later
    checks.append(mu(lambda n: n - 7, lo=2) == 7)
    # never zero returns None
    checks.append(mu(lambda n: 1) is None)
    # root of n^2 - 9 at n=3
    checks.append(mu(lambda n: n * n - 9 if n * n >= 9 else 1) == 3)
    # first even n where n(n+1) == 12*13 -> n=12
    checks.append(mu(lambda n: n * (n + 1) - 156) == 12)
    return float(sum(checks) / len(checks))


def bench_mu_recursion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mu_recursion": _bench_mu_recursion(seed)}
