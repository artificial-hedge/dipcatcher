"""SYNTHETIC trampoline tail-call optimization.

CPS-style functions return ('call', thunk) for tail positions and
('done', value) otherwise; the trampoline loops — recursion depth
>10⁵ without stack overflow, results equal to naive (bounded) eval.
"""

from __future__ import annotations

import random
import sys


def tramp(f, *args):
    v = f(*args)
    while isinstance(v, tuple) and v[0] == "call":
        v = v[1]()
    return v[1]


def _sum_tail(n: int, acc: int = 0):
    if n == 0:
        return ("done", acc)
    return ("call", lambda: _sum_tail(n - 1, acc + n))


def _fib_tail(n: int, a: int = 0, b: int = 1):
    if n == 0:
        return ("done", a)
    return ("call", lambda: _fib_tail(n - 1, b, a + b))


def _fib_ref(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def bench_tail_call_tramp(seed: int = 20261231 + 483) -> dict[str, float]:
    rng = random.Random(seed)
    deep = correct = beyond = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(10**4, 10**5)
        deep += int(tramp(_sum_tail, n) == n * (n + 1) // 2)
        m = rng.randrange(10, 200)
        correct += int(tramp(_fib_tail, m) == _fib_ref(m))
        # beyond default recursion limit, no RecursionError
        lim = sys.getrecursionlimit()
        ok = True
        try:
            tramp(_sum_tail, lim * 4)
        except RecursionError:
            ok = False
        beyond += int(ok)
    return {
        "synthetic_deep_recursion": float(deep / trials),
        "synthetic_matches_reference": float(correct / trials),
        "synthetic_beyond_recursionlimit": float(beyond / trials),
    }
