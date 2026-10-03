"""Dependent types lite: Pi/Sigma types, dependent application where the
codomain instantiates with the argument's value, length-indexed Vec
(SYNTHETIC bench only)."""

from __future__ import annotations

from dataclasses import dataclass

_SEED = 20261231 + 995


@dataclass(frozen=True)
class Ty:
    tag: str
    args: tuple = ()


NAT = Ty("nat")
VEC = lambda n: Ty("vec", (n,))  # noqa: E731
PI = lambda v, a, b: Ty("pi", (v, a, b))  # noqa: E731
SIG = lambda v, a, b: Ty("sig", (v, a, b))  # noqa: E731


@dataclass(frozen=True)
class Lam:
    param: str
    body: object
    env: dict


def eval_(e, env):
    tag = e[0]
    if tag == "var":
        return env[e[1]]
    if tag == "lit":
        return ("lit", e[1])
    if tag == "lam":
        return Lam(e[1], e[2], dict(env))
    if tag == "app":
        f = eval_(e[1], env)
        a = eval_(e[2], env)
        env2 = dict(f.env)
        env2[f.param] = a
        return eval_(f.body, env2)
    if tag == "vec":
        return ("vec", tuple(eval_(x, env) for x in e[1]))
    if tag == "vappend":
        v1 = eval_(e[1], env)[1]
        v2 = eval_(e[2], env)[1]
        return ("vec", v1 + v2)
    if tag == "add":
        return ("lit", eval_(e[1], env)[1] + eval_(e[2], env)[1])
    raise ValueError(tag)


def subst_ty(ty: Ty, var: str, val) -> Ty:
    if ty.tag == "vec":
        return VEC(_subst_expr(ty.args[0], var, val))
    if ty.tag == "pi":
        v, a, b = ty.args
        return PI(v, subst_ty(a, var, val), subst_ty(b, var, val) if v != var else b)
    return ty


def _subst_expr(e, var, val):
    if isinstance(e, tuple):
        if e[0] == "var" and e[1] == var:
            return val
        return tuple(_subst_expr(x, var, val) for x in e)
    return e


def nat_of(v) -> int:
    return int(v[1])


def infer(e, ctx):
    tag = e[0]
    if tag == "lit":
        return NAT
    if tag == "var":
        return ctx[e[1]]
    if tag == "vec":
        for x in e[1]:
            check(x, NAT, ctx)
        return VEC(("lit", len(e[1])))
    if tag == "vappend":
        t1 = infer(e[1], ctx)
        t2 = infer(e[2], ctx)
        if t1.tag != "vec" or t2.tag != "vec":
            raise TypeError("vappend on non-vec")
        n1 = nat_of(eval_(t1.args[0], ctx.get("_env", {})))
        n2 = nat_of(eval_(t2.args[0], ctx.get("_env", {})))
        return VEC(("lit", n1 + n2))
    if tag == "app":
        ft = infer(e[1], ctx)
        if ft.tag != "pi":
            raise TypeError("apply non-Pi")
        v, a, b = ft.args
        check(e[2], a, ctx)
        arg = eval_(e[2], ctx.get("_env", {}))
        return subst_ty(
            b,
            v,
            arg if arg[0] == "lit" else ("lit", nat_of(arg)) if isinstance(arg, tuple) else arg,
        )
    raise TypeError(tag)


def check(e, ty: Ty, ctx) -> None:
    if e[0] == "lam" and ty.tag == "pi":
        v, a, b = ty.args
        c2 = dict(ctx)
        c2[e[1]] = a
        check(e[2], b, c2)
        return
    got = infer(e, ctx)
    if not _ty_eq(got, ty, ctx):
        raise TypeError(f"{got} != {ty}")


def _ty_eq(t1: Ty, t2: Ty, ctx) -> bool:
    if t1.tag != t2.tag:
        return False
    if t1.tag == "vec":
        n1 = nat_of(eval_(t1.args[0], ctx.get("_env", {})))
        n2 = nat_of(eval_(t2.args[0], ctx.get("_env", {})))
        return n1 == n2
    if t1.tag == "pi":
        v1, a1, b1 = t1.args
        v2, a2, b2 = t2.args
        b2s = subst_ty(b2, v2, ("var", v1)) if v1 != v2 else b2
        return _ty_eq(a1, a2, ctx) and _ty_eq(b1, b2s, ctx)
    return t1 == t2


def bench_dep_types(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    checks.append(infer(("vec", (("lit", 1), ("lit", 2))), {}) == VEC(("lit", 2)))
    v3 = ("vec", (("lit", 1), ("lit", 2), ("lit", 3)))
    app_ty = infer(("vappend", ("vec", (("lit", 1),)), v3), {})
    checks.append(app_ty == VEC(("lit", 4)))
    id_vec = ("lam", "v", ("var", "v"))
    c: dict = {"_env": {}}
    check(id_vec, PI("v", VEC(("lit", 4)), VEC(("lit", 4))), c)
    checks.append(True)
    try:
        check(id_vec, PI("v", VEC(("lit", 3)), VEC(("lit", 4))), c)
        checks.append(False)
    except TypeError:
        checks.append(True)
    try:
        check(("vec", (("lit", 1),)), VEC(("lit", 2)), {})
        checks.append(False)
    except TypeError:
        checks.append(True)
    ctx = {"f": PI("n", NAT, VEC(("var", "n"))), "_env": {}}
    rt = infer(("app", ("var", "f"), ("lit", 5)), ctx)
    checks.append(rt == VEC(("lit", 5)))
    return {"synthetic_dep_types": float(sum(checks) / len(checks))}
