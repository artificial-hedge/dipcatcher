"""Bidirectional type checker for STLC + pairs, lets, literals
(SYNTHETIC bench only)."""

from __future__ import annotations

from dataclasses import dataclass

_SEED = 20261231 + 993


@dataclass(frozen=True)
class Ty:
    tag: str
    a: Ty | None = None
    b: Ty | None = None


NAT, BOOL = Ty("nat"), Ty("bool")


def arrow(a: Ty, b: Ty) -> Ty:
    return Ty("->", a, b)


def prod(a: Ty, b: Ty) -> Ty:
    return Ty("*", a, b)


Expr = tuple
Ctx = dict[str, Ty]


class TypeError_(Exception):
    pass


def infer(e: Expr, ctx: Ctx) -> Ty:
    tag = e[0]
    if tag == "lit":
        return BOOL if isinstance(e[1], bool) else NAT
    if tag == "var":
        if e[1] not in ctx:
            raise TypeError_(f"unbound {e[1]}")
        return ctx[e[1]]
    if tag == "lam":
        _, name, ty, body = e
        c2 = dict(ctx)
        c2[name] = ty
        return arrow(ty, infer(body, c2))
    if tag == "app":
        ft = infer(e[1], ctx)
        if ft.tag != "->":
            raise TypeError_("apply non-function")
        check(e[2], ft.a, ctx)  # type: ignore[arg-type]
        return ft.b  # type: ignore[return-value]
    if tag == "pair":
        return prod(infer(e[1], ctx), infer(e[2], ctx))
    if tag in ("fst", "snd"):
        pt = infer(e[1], ctx)
        if pt.tag != "*":
            raise TypeError_("project non-pair")
        sub = pt.a if tag == "fst" else pt.b
        if sub is None:
            raise TypeError_("empty product")
        return sub
    if tag == "let":
        _, name, val, body = e
        vt = infer(val, ctx)
        c2 = dict(ctx)
        c2[name] = vt
        return infer(body, c2)
    if tag == "ann":
        aty = e[2]
        if not isinstance(aty, Ty):
            raise TypeError_("bad annotation")
        check(e[1], aty, ctx)
        return aty
    if tag == "if":
        check(e[1], BOOL, ctx)
        t1 = infer(e[2], ctx)
        check(e[3], t1, ctx)
        return t1
    if tag == "add":
        check(e[1], NAT, ctx)
        check(e[2], NAT, ctx)
        return NAT
    raise TypeError_(f"unknown expr {tag}")


def check(e: Expr, ty: Ty, ctx: Ctx) -> None:
    if e[0] == "lam" and ty.tag == "->":
        _, name, _, body = e
        c2 = dict(ctx)
        c2[name] = ty.a  # type: ignore[assignment]
        check(body, ty.b, c2)  # type: ignore[arg-type]
        return
    if e[0] == "pair" and ty.tag == "*":
        check(e[1], ty.a, ctx)  # type: ignore[arg-type]
        check(e[2], ty.b, ctx)  # type: ignore[arg-type]
        return
    if e[0] == "if":
        check(e[1], BOOL, ctx)
        check(e[2], ty, ctx)
        check(e[3], ty, ctx)
        return
    got = infer(e, ctx)
    if got != ty:
        raise TypeError_(f"expected {ty}, got {got}")


def bench_bidirectional_tc(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    idnat = ("lam", "x", NAT, ("var", "x"))
    checks.append(infer(idnat, {}) == arrow(NAT, NAT))
    const = (
        "lam",
        "x",
        NAT,
        ("lam", "y", BOOL, ("var", "x")),
    )
    checks.append(infer(const, {}) == arrow(NAT, arrow(BOOL, NAT)))
    app = ("app", idnat, ("lit", 5))
    checks.append(infer(app, {}) == NAT)
    p = ("pair", ("lit", 3), ("lit", True))
    checks.append(infer(("fst", p), {}) == NAT)
    checks.append(infer(("snd", p), {}) == BOOL)
    try:
        infer(("app", ("lit", 1), ("lit", 2)), {})
        checks.append(False)
    except TypeError_:
        checks.append(True)
    try:
        check(("lit", True), NAT, {})
        checks.append(False)
    except TypeError_:
        checks.append(True)
    lets = ("let", "f", idnat, ("app", ("var", "f"), ("lit", 7)))
    checks.append(infer(lets, {}) == NAT)
    try:
        infer(("var", "free"), {})
        checks.append(False)
    except TypeError_:
        checks.append(True)
    ite = ("if", ("lit", True), ("lit", 1), ("lit", 2))
    checks.append(infer(ite, {}) == NAT)
    return {"synthetic_bidirectional_tc": float(sum(checks) / len(checks))}
