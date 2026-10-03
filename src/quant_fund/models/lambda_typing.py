"""Simply-typed lambda calculus: bidirectional type checker (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

Term = Any  # ("var",name)|("lam",name,ty,body)|("app",f,x)|("lit",int)
Ty = Any  # "int" | ("->",A,B)


def infer(ctx: dict[str, Ty], t: Term) -> Ty:
    if t[0] == "var":
        if t[1] not in ctx:
            raise TypeError(f"unbound {t[1]}")
        return ctx[t[1]]
    if t[0] == "lit":
        return "int"
    if t[0] == "lam":
        _, x, ty, body = t
        return ("->", ty, infer({**ctx, x: ty}, body))
    if t[0] == "app":
        tf = infer(ctx, t[1])
        if not (isinstance(tf, tuple) and tf[0] == "->"):
            raise TypeError("app of non-function")
        check(ctx, t[2], tf[1])
        return tf[2]
    raise ValueError(t)


def check(ctx: dict[str, Ty], t: Term, ty: Ty) -> None:
    if t[0] == "lam" and isinstance(ty, tuple) and ty[0] == "->":
        _, x, xty, body = t
        if xty != ty[1]:
            raise TypeError("binder annotation mismatch")
        check({**ctx, x: xty}, body, ty[2])
        return
    got = infer(ctx, t)
    if got != ty:
        raise TypeError(f"{got} != {ty}")


def well_typed(t: Term) -> bool:
    try:
        infer({}, t)
        return True
    except (TypeError, ValueError):
        return False


def _bench_lambda_typing(seed: int = 0) -> float:
    checks = []
    id_int = ("lam", "x", "int", ("var", "x"))
    checks.append(infer({}, id_int) == ("->", "int", "int"))
    checks.append(infer({}, ("app", id_int, ("lit", 3))) == "int")
    const = ("lam", "x", "int", ("lam", "y", "int", ("var", "x")))
    checks.append(infer({}, const) == ("->", "int", ("->", "int", "int")))
    # self application ill-typed
    checks.append(not well_typed(("lam", "x", "int", ("app", ("var", "x"), ("var", "x")))))
    checks.append(not well_typed(("app", ("lit", 1), ("lit", 2))))
    checks.append(well_typed(("app", ("app", const, ("lit", 1)), ("lit", 2))))
    return float(sum(checks) / len(checks))


def bench_lambda_typing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lambda_typing": _bench_lambda_typing(seed)}
