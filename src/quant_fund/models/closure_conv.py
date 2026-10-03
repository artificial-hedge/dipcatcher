"""SYNTHETIC closure conversion: λ → (env, code) pairs.

Free-variable analysis builds explicit environments; converted
evaluation must equal direct lexical-scope evaluation, including
returned (escaping) closures.
"""

from __future__ import annotations

import random


def _free(e: tuple, bound: set[str]) -> set[str]:
    op = e[0]
    if op == "var":
        return set() if e[1] in bound else {e[1]}
    if op == "lit":
        return set()
    if op == "lam":
        return _free(e[2], bound | {e[1]})
    if op == "app":
        return _free(e[1], bound) | _free(e[2], bound)
    return _free(e[1], bound) | _free(e[2], bound)  # binop


def conv(e: tuple, env: dict[str, int]) -> tuple:
    """Return converted closure: ('closure', freevars_tuple, body, env)."""
    fv = sorted(_free(e, {e[1]} if e[0] == "lam" else set()))
    return ("closure", tuple(fv), e, {k: env[k] for k in fv if k in env})


def eval_cc(e: tuple, env: dict[str, int]) -> int | tuple:
    op = e[0]
    if op == "lit":
        return int(e[1])
    if op == "var":
        return env[e[1]]
    if op in ("add", "sub"):
        lo = eval_cc(e[1], env)
        ro = eval_cc(e[2], env)
        assert isinstance(lo, int) and isinstance(ro, int)
        return lo + ro if op == "add" else lo - ro
    if op == "lam":
        return conv(e, env)
    if op == "app":
        fn = eval_cc(e[1], env)
        arg = eval_cc(e[2], env)
        assert isinstance(fn, tuple) and fn[0] == "closure"
        _, _, body, fenv = fn
        assert isinstance(arg, int)
        inner = dict(fenv)
        inner[body[1]] = arg
        return eval_cc(body[2], inner)
    raise ValueError(op)


def bench_closure_conv(seed: int = 20261231 + 482) -> dict[str, float]:
    rng = random.Random(seed)
    eval_ok = capture = return_ok = 0
    trials = 60
    for _ in range(trials):
        y, c = rng.randrange(-9, 10), rng.randrange(-9, 10)
        # (λx. x + c) applied to y  — c is free, captured in env
        lam = ("lam", "x", ("add", ("var", "x"), ("var", "c")))
        e = ("app", lam, ("lit", y))
        direct = y + c
        got = eval_cc(e, {"c": c})
        eval_ok += int(got == direct)
        # escaping closure: capture env must hold c
        clo = eval_cc(lam, {"c": c})
        assert isinstance(clo, tuple)
        capture += int(clo[3] == {"c": c})
        # apply later from another env
        _, _, body, fenv = clo
        inner = dict(fenv)
        inner["x"] = y
        return_ok += int(eval_cc(body[2], inner) == direct)
    return {
        "synthetic_converted_eval_correct": float(eval_ok / trials),
        "synthetic_env_captured": float(capture / trials),
        "synthetic_escaping_closure": float(return_ok / trials),
    }
