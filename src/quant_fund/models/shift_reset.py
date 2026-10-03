"""Delimited continuations: reset / shift via CPS evaluation.

reset(e) evaluates e in continuation-passing style; shift(f) captures
the current continuation up to the nearest reset and passes it to f
as a first-class function. Implemented as a small CPS interpreter —
expressions are tuples over a tiny arithmetic/combinator language.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SEED = 20261231 + 1025

Cont = Callable[[Any], Any]


def _cps(e: Any, k: Cont) -> Any:
    tag = e[0]
    if tag == "lit":
        return k(e[1])
    if tag == "var":
        raise ValueError("unbound")
    if tag in ("add", "sub", "mul"):
        return _cps(
            e[1], lambda a: _cps(e[2], lambda b: k({"add": a + b, "sub": a - b, "mul": a * b}[tag]))
        )
    if tag == "reset":
        return k(_cps(e[1], lambda v: v))
    if tag == "shift":
        # capture k: run f's body with a reified continuation; result is the
        # nearest reset's answer
        f = e[1]
        return _cps(f(lambda v: lambda kk: kk(k(v))), lambda v: v)
    raise ValueError(e)


def eval_reset(e: Any) -> Any:
    return _cps(e, lambda v: v)


def bench_shift_reset(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # 1 + reset(2 + shift(k. k(3))) = 1 + (k(3) where k = 2+_) = 1 + 5 = 6
    e = (
        "add",
        ("lit", 1),
        (
            "reset",
            (
                "add",
                ("lit", 2),
                ("shift", lambda k: ("app", k, ("lit", 3))),
            ),
        ),
    )
    # need "app" node to call reified k — extend via wrapper
    checks.append(_app_eval(e) == 6)
    # shift returns twice: reset(shift(k. k(k(2)))) + 10 => k(k(2)) where k=id+10 => 22? classic = 12+... compute
    e2 = (
        "add",
        ("lit", 10),
        ("reset", ("shift", lambda k: ("app", k, ("app", k, ("lit", 2))))),
    )
    # k captures only the in-reset context (empty) -> k(k(2)) = 2 -> 10+2 = 12
    checks.append(_app_eval(e2) == 12)
    # no shift: reset is transparent
    checks.append(eval_reset(("add", ("lit", 2), ("reset", ("lit", 3)))) == 5)
    # shift abort: shift(k. 99) discards ctx -> reset gives 99
    checks.append(
        eval_reset(("reset", ("add", ("lit", 1), ("shift", lambda k: ("lit", 99))))) == 99
    )
    return {"synthetic_shift_reset": float(sum(checks)) / len(checks)}


def _app_eval(e: Any) -> Any:
    """CPS eval extended with ("app", kthunk, arg) applying a reified cont."""

    def cps(x: Any, k: Cont) -> Any:
        tag = x[0]
        if tag == "lit":
            return k(x[1])
        if tag in ("add", "sub", "mul"):
            return cps(
                x[1],
                lambda a: cps(x[2], lambda b: k({"add": a + b, "sub": a - b, "mul": a * b}[tag])),
            )
        if tag == "reset":
            return k(cps(x[1], lambda v: v))
        if tag == "shift":
            f = x[1]
            return cps(f(lambda v: lambda kk: kk(k(v))), lambda v: v)
        if tag == "app":
            # x[1] builds a fn that when applied to a value v returns a
            # computation expecting kk; we feed identity to get value
            fn = x[1]
            return cps(x[2], lambda a: k(fn(a)(lambda z: z)))
        raise ValueError(x)

    return cps(e, lambda v: v)
