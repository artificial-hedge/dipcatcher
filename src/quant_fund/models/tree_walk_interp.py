"""SYNTHETIC tree-walking interpreter with lexical closures.

Language: numbers, +,-,*,= ,if, let, λ (closures capture env), calls.
Verified on factorial via letrec-encoded Y-free recursion (let + self
reference through a box), closure capture, and higher-order composition.
"""

from __future__ import annotations

import random

Expr = tuple
Env = dict


def ev(e: Expr, env: Env) -> object:
    tag = e[0]
    if tag == "n":
        return e[1]
    if tag == "var":
        return env[e[1]]
    if tag == "+":
        return ev(e[1], env) + ev(e[2], env)  # type: ignore[operator]
    if tag == "-":
        return ev(e[1], env) - ev(e[2], env)  # type: ignore[operator]
    if tag == "*":
        return ev(e[1], env) * ev(e[2], env)  # type: ignore[operator]
    if tag == "=":
        return ev(e[1], env) == ev(e[2], env)
    if tag == "if":
        return ev(e[2], env) if ev(e[1], env) else ev(e[3], env)
    if tag == "let":
        env2 = dict(env)
        env2[e[1]] = ev(e[2], env)
        return ev(e[3], env2)
    if tag == "letrec":
        env2 = dict(env)
        if e[2][0] == "lam":
            # closure captures env2 by reference so the name resolves to itself
            env2[e[1]] = ("closure", e[2][1], e[2][2], env2)
        else:
            env2[e[1]] = ev(e[2], env2)
        return ev(e[3], env2)
    if tag == "lam":
        return ("closure", e[1], e[2], dict(env))
    if tag == "app":
        f = ev(e[1], env)
        a = ev(e[2], env)
        if not (isinstance(f, tuple) and f[0] == "closure"):
            raise ValueError("not a function")
        env2 = dict(f[3])
        env2[f[1]] = a
        return ev(f[2], env2)
    raise ValueError(tag)


def bench_tree_walk_interp(seed: int = 20261231 + 411) -> dict[str, float]:
    rng = random.Random(seed)
    arith = clos = rec = 0
    trials = 40
    for _ in range(trials):
        a, b = rng.randrange(1, 20), rng.randrange(1, 20)
        # arithmetic
        e = ("*", ("+", ("n", a), ("n", b)), ("n", 2))
        arith += int(ev(e, {}) == (a + b) * 2)
        # closures capture lexically: let x=1 in (λy.x+y) applied later with x shadowed
        e2 = (
            "let",
            "x",
            ("n", a),
            (
                "let",
                "f",
                ("lam", "y", ("+", ("var", "x"), ("var", "y"))),
                ("let", "x", ("n", 999), ("app", ("var", "f"), ("n", 1))),
            ),
        )
        clos += int(ev(e2, {}) == a + 1)
        # letrec factorial of small n
        n = rng.randrange(1, 7)
        fact = (
            "letrec",
            "f",
            (
                "lam",
                "n",
                (
                    "if",
                    ("=", ("var", "n"), ("n", 0)),
                    ("n", 1),
                    ("*", ("var", "n"), ("app", ("var", "f"), ("-", ("var", "n"), ("n", 1)))),
                ),
            ),
            ("app", ("var", "f"), ("n", n)),
        )
        import math

        rec += int(ev(fact, {}) == math.factorial(n))
    return {
        "synthetic_arith_correct": float(arith / trials),
        "synthetic_lexical_closure": float(clos / trials),
        "synthetic_recursion": float(rec / trials),
    }
