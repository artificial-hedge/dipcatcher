"""SKI combinator calculus: normal-order reduction + derived combinators (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

Term = Any  # "S"|"K"|"I"|("app",f,x)|("atom",name)


def app(*ts: Term) -> Term:
    t = ts[0]
    for x in ts[1:]:
        t = ("app", t, x)
    return t


def reduce(t: Term, fuel: int = 4000) -> Term:
    for _ in range(fuel):
        t2 = step_all(t)
        if t2 == t:
            return t
        t = t2
    raise RecursionError("no normal form")


def step_all(t: Term) -> Term:
    """One leftmost-outermost pass: returns term after reducing all outermost redexes once."""
    if isinstance(t, str) or (isinstance(t, tuple) and t[0] == "atom"):
        return t
    f, x = t[1], t[2]
    # S f g x -> f x (g x); K a b -> a; I x -> x
    while isinstance(f, tuple) and f[0] == "app":
        f = ("app", reduce_inner(f[1]), f[2])
        break
    if f == "I":
        return x
    if isinstance(f, tuple) and f[0] == "app" and f[1] == "K":
        return f[2]
    if (
        isinstance(f, tuple)
        and f[0] == "app"
        and isinstance(f[1], tuple)
        and f[1][0] == "app"
        and f[1][1] == "S"
    ):
        return app(app(f[1][2], x), app(f[2], x))
    return ("app", step_all(f), step_all(x))


def reduce_inner(t: Term) -> Term:
    return t


def eq(t1: Term, t2: Term) -> bool:
    return bool(reduce(t1) == reduce(t2))


def _bench_ski_combinator(seed: int = 0) -> float:
    a = ("atom", "a")
    b = ("atom", "b")
    checks = []
    checks.append(eq(app("I", a), a))
    checks.append(eq(app("K", a, b), a))
    # SKK = identity
    checks.append(eq(app("S", "K", "K", a), a))
    # T = S(K(SI))K is the swap combinator: T x y = y x
    comb_t = app("S", app("K", app("S", "I")), "K")
    checks.append(eq(app(comb_t, ("atom", "x"), ("atom", "y")), app(("atom", "y"), ("atom", "x"))))
    # B = S(KS)K is composition: B f g x = f(g x)
    comb_b = app("S", app("K", "S"), "K")
    checks.append(
        eq(
            app(comb_b, ("atom", "f"), ("atom", "g"), ("atom", "x")),
            app(("atom", "f"), app(("atom", "g"), ("atom", "x"))),
        )
    )
    return float(sum(checks) / len(checks))


def bench_ski_combinator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ski_combinator": _bench_ski_combinator(seed)}
