"""Partial evaluation: specialize a tiny expression on known inputs (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 722

# expr: ("const", v) | ("var", name) | ("add"|"mul", l, r)
Expr = tuple


def peval(expr: Expr, env: dict[str, int]) -> Expr:
    tag = expr[0]
    if tag == "const":
        return expr
    if tag == "var":
        return ("const", env[expr[1]]) if expr[1] in env else expr
    lft = peval(expr[1], env)
    r = peval(expr[2], env)
    if lft[0] == "const" and r[0] == "const":
        v = lft[1] + r[1] if tag == "add" else lft[1] * r[1]
        return ("const", v)
    if tag == "mul":
        if lft == ("const", 0) or r == ("const", 0):
            return ("const", 0)
        if lft == ("const", 1):
            return r
        if r == ("const", 1):
            return lft
    if tag == "add":
        if lft == ("const", 0):
            return r
        if r == ("const", 0):
            return lft
    return (tag, lft, r)


def run_expr(expr: Expr, env: dict[str, int]) -> int:
    tag = expr[0]
    if tag == "const":
        return int(expr[1])
    if tag == "var":
        return env[expr[1]]
    lft = run_expr(expr[1], env)
    r = run_expr(expr[2], env)
    return lft + r if tag == "add" else lft * r


def bench_partial_eval(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        # random binary tree depth ≤ 3 over vars x,y
        def gen(d: int) -> Expr:
            if d == 0 or rng.rand() < 0.3:
                if rng.rand() < 0.5:
                    return ("const", int(rng.randint(-5, 5)))
                return ("var", rng.choice(["x", "y"]))
            op = "add" if rng.rand() < 0.5 else "mul"
            return (op, gen(d - 1), gen(d - 1))

        e = gen(3)
        known = {"x": int(rng.randint(-3, 3))}
        resid = peval(e, known)
        # residual must be a valid expr and preserve value on all y
        good = all(
            run_expr(resid, {"y": y}) == run_expr(e, {**known, "y": y}) for y in range(-3, 4)
        )

        # specialization should not grow the tree
        def size(t: Expr) -> int:
            return 1 if t[0] in ("const", "var") else 1 + size(t[1]) + size(t[2])

        ok += float(good and size(resid) <= size(e))
    return {"synthetic_peval_preserves": ok / trials}
