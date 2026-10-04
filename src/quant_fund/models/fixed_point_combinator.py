"""Applicative fixed points over total functions (SYNTHETIC)."""

from __future__ import annotations


def fix(f):
    """Bottom-of-domain fixed point via iteration from a default seed:
    f maps partial fn -> partial fn; iterate from everywhere-0."""

    def g(x):
        return 0

    cur = g
    for _ in range(64):
        nxt = f(cur)
        if all(nxt(x) == cur(x) for x in range(12)):
            return nxt
        cur = nxt
    return cur


def _bench_fixed_point_combinator(seed: int = 0) -> float:
    checks = []

    # fixed point of F(g)(n) = 1 if n==0 else n*g(n-1) is factorial
    def fact_fun(g):
        return lambda n: 1 if n == 0 else n * g(n - 1)

    fact = fix(fact_fun)
    checks.append(fact(5) == 120)
    checks.append(fact(0) == 1)

    # fixed point of F(g)(n) = max(g(n), n) is the identity
    def lift(g):
        return lambda n: max(g(n), n)

    f2 = fix(lift)
    checks.append(f2(3) == 3)
    checks.append(f2(7) == 7)

    # constant functional has immediate fixed point
    def const(g):
        return lambda n: 7

    checks.append(fix(const)(100) == 7)

    # fibonacci functional
    def fib_fun(g):
        return lambda n: n if n < 2 else g(n - 1) + g(n - 2)

    fib = fix(fib_fun)
    checks.append(fib(10) == 55)
    return float(sum(checks) / len(checks))


def bench_fixed_point_combinator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fixed_point_combinator": _bench_fixed_point_combinator(seed)}
