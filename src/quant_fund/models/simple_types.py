"""SYNTHETIC simply-typed lambda calculus checker (no inference).

Types: Int, Bool, Int→Int etc. Terms carry annotations on λ. Verify:
well-typed terms pass, known bad terms rejected, function application
respects domain types.
"""

from __future__ import annotations

import random

Type = tuple  # ("Int",) ("Bool",) ("->",A,B)
Expr = tuple  # ("n",v) ("b",v) ("var",x) ("lam",x,T,e) ("app",f,a) ("if",c,t,e)


def check(e: Expr, env: dict[str, Type]) -> Type:
    tag = e[0]
    if tag == "n":
        return ("Int",)
    if tag == "b":
        return ("Bool",)
    if tag == "var":
        t: Type = env[e[1]]
        return t
    if tag == "lam":
        env2 = dict(env)
        env2[e[1]] = e[2]
        return ("->", e[2], check(e[3], env2))
    if tag == "app":
        tf = check(e[1], env)
        ta = check(e[2], env)
        if tf[0] != "->" or tf[1] != ta:
            raise ValueError("domain mismatch")
        out: Type = tf[2]
        return out
    if tag == "if":
        tc = check(e[1], env)
        if tc != ("Bool",):
            raise ValueError("cond not bool")
        t1, t2 = check(e[2], env), check(e[3], env)
        if t1 != t2:
            raise ValueError("branch mismatch")
        return t1
    raise ValueError(tag)


def bench_simple_types(seed: int = 20261231 + 415) -> dict[str, float]:
    rng = random.Random(seed)
    good = bad = arrow = 0
    trials = 40
    for _ in range(trials):
        INT = ("Int",)
        # λx:Int. x  has type Int→Int
        t = check(("lam", "x", INT, ("var", "x")), {})
        good += int(t == ("->", INT, INT))
        # (λx:Int. x) applied to bool → reject
        try:
            check(("app", ("lam", "x", INT, ("var", "x")), ("b", True)), {})
            bad += 0
        except ValueError:
            bad += 1
        # higher-order: λf:Int→Int. λx:Int. (f x)
        t2 = check(
            ("lam", "f", ("->", INT, INT), ("lam", "x", INT, ("app", ("var", "f"), ("var", "x")))),
            {},
        )
        arrow += int(t2 == ("->", ("->", INT, INT), ("->", INT, INT)))
        _ = rng.random()
    return {
        "synthetic_typed_accept": float(good / trials),
        "synthetic_typed_reject": float(bad / trials),
        "synthetic_arrow_types": float(arrow / trials),
    }
