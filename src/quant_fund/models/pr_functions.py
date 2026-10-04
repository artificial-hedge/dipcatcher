"""Primitive-recursive function algebra (zero/succ/proj/compose/recurse, SYNTHETIC)."""

from __future__ import annotations

PRF = tuple  # ("z",) | ("s",) | ("p",i,n) | ("c",g,[f1..fk]) | ("r",g,h)


def eval_prf(f: PRF, args: tuple[int, ...], fuel: int = 10000) -> int:
    if fuel <= 0:
        raise RecursionError("fuel")
    tag = f[0]
    if tag == "z":
        return 0
    if tag == "s":
        return args[0] + 1
    if tag == "p":
        _, i, n = f
        return args[i] if i < len(args) else 0
    if tag == "c":
        _, g, fs = f
        return eval_prf(g, tuple(eval_prf(x, args, fuel - 1) for x in fs), fuel - 1)
    if tag == "r":
        _, g, h = f
        n = args[-1]
        base = args[:-1]
        v = eval_prf(g, base, fuel - 1)
        for k in range(n):
            v = eval_prf(h, (k, v) + base, fuel - 1)
        return v
    raise ValueError(tag)


ZERO = ("z",)
SUCC = ("s",)
ADD = ("r", ("p", 0, 1), ("c", ("s",), [("p", 1, 3)]))
MUL = ("r", ("z",), ("c", ADD, [("p", 1, 3), ("p", 2, 3)]))
PRED = ("r", ("z",), ("p", 0, 2))
EXP = ("r", ("c", ("s",), [("z",)]), ("c", MUL, [("p", 1, 3), ("p", 2, 3)]))


def ack(m: int, n: int) -> int:
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))


def _bench_pr_functions(seed: int = 0) -> float:
    checks = []
    checks.append(eval_prf(ADD, (3, 4)) == 7)
    checks.append(eval_prf(MUL, (3, 4)) == 12)
    checks.append(eval_prf(PRED, (5,)) == 4)
    checks.append(eval_prf(EXP, (2, 3)) == 8)
    checks.append(ack(2, 3) == 9)
    # Ackermann grows faster than any fixed PR stack: A(3,3)=61 > exp tower of height 2
    checks.append(ack(3, 3) == 61 and ack(3, 3) > eval_prf(EXP, (2, 5)))
    return sum(checks) / len(checks)


def bench_pr_functions(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pr_functions": _bench_pr_functions(seed)}
