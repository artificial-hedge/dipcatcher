"""SYNTHETIC macro expansion + desugaring.

Macros: (when c e) → (if c e ()), (let* ((x v)...) e) → nested let,
(my-and a b) → (if a b #f). Expansion evaluated by a tiny interpreter;
verified against manual desugar results.
"""

from __future__ import annotations

import random

Expr = object


def expand(e: object) -> object:
    if not isinstance(e, list) or not e:
        return e
    head = e[0]
    if head == "when":
        return ["if", expand(e[1]), expand(e[2]), None]
    if head == "let*":
        binds = e[1]
        body = expand(e[2])
        for name, val in reversed(binds):
            body = ["let", name, expand(val), body]
        return body
    if head == "and":
        if len(e) == 3:
            return ["if", expand(e[1]), expand(e[2]), False]
        return True
    return [expand(x) for x in e]


def interp(e: object, env: dict) -> object:
    if isinstance(e, bool) or e is None or isinstance(e, (int, float)):
        return e
    if isinstance(e, str):
        return env[e]
    if not isinstance(e, list):
        raise ValueError(e)
    head = e[0]
    if head == "if":
        return interp(e[2], env) if interp(e[1], env) else interp(e[3], env)
    if head == "let":
        env2 = dict(env)
        env2[e[1]] = interp(e[2], env)
        return interp(e[3], env2)
    if head == "+":
        return interp(e[1], env) + interp(e[2], env)  # type: ignore[operator]
    if head == "*":
        return interp(e[1], env) * interp(e[2], env)  # type: ignore[operator]
    if head == ">":
        return interp(e[1], env) > interp(e[2], env)  # type: ignore[operator]
    raise ValueError(head)


def bench_macro_expand(seed: int = 20261231 + 413) -> dict[str, float]:
    rng = random.Random(seed)
    when_ok = let_ok = and_ok = 0
    trials = 40
    for _ in range(trials):
        a, b = rng.randrange(0, 10), rng.randrange(0, 10)
        # (when (> a b) (+ a b)) == if a>b → a+b else None
        e = ["when", [">", a, b], ["+", a, b]]
        exp = a + b if a > b else None
        when_ok += int(interp(expand(e), {}) == exp)
        # let* chains
        e2 = ["let*", [["x", a], ["y", ["+", "x", b]]], ["*", "x", "y"]]
        let_ok += int(interp(expand(e2), {}) == a * (a + b))
        # and short-circuit: (and #f x) never touches unbound x;
        # (and #t ...) must evaluate (error proves it reached x)
        try:
            got = interp(expand(["and", False, "x"]), {})
            and_ok += int(got is False)
        except (ValueError, KeyError):
            and_ok += 0
        try:
            interp(expand(["and", True, "x"]), {})
            and_ok += 0
        except (ValueError, KeyError):
            and_ok += 1
    return {
        "synthetic_when_correct": float(when_ok / trials),
        "synthetic_let_star_correct": float(let_ok / trials),
        "synthetic_and_lazy": float(and_ok / (trials * 2)),
    }
