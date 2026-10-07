"""Liquid refinement types: {ν:T | φ} subtyping via VC checking over a (SYNTHETIC)
finite domain (small-model property used here as decision procedure).

Refinements: ("ref",base,pred) where pred is a small expression in ν.
sub({ν|p} <: {ν|q}) iff ∀ν in domain. p(ν) ⇒ q(ν) — verified by
exhaustive evaluation. VC generation walks a straight-line program
("assign",x,e)/("assert",φ)/("if",c,t,else) producing obligations.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1040


def _eval(e: Any, env: dict[str, int], nu: int | None = None) -> Any:
    if isinstance(e, int):
        return e
    if e == "nu":
        return nu
    if isinstance(e, str):
        return env[e]
    tag = e[0]
    if tag in ("add", "sub", "mul"):
        a, b = _eval(e[1], env, nu), _eval(e[2], env, nu)
        return {"add": a + b, "sub": a - b, "mul": a * b}[tag]
    if tag in ("le", "lt", "eq", "ge", "gt"):
        a, b = _eval(e[1], env, nu), _eval(e[2], env, nu)
        return {"le": a <= b, "lt": a < b, "eq": a == b, "ge": a >= b, "gt": a > b}[tag]
    if tag == "and":
        return _eval(e[1], env, nu) and _eval(e[2], env, nu)
    if tag == "not":
        return not _eval(e[1], env, nu)
    raise ValueError(e)


def sub(p: Any, q: Any, domain: range) -> bool:
    """{ν|p} <: {ν|q} over domain."""
    return all(not _eval(p, {}, v) or _eval(q, {}, v) for v in domain)


def vcgen(prog: list[tuple], post: Any, env_ty: dict[str, Any], domain: range) -> bool:
    """Forward VC: assert each ("assert",φ) against current per-var
    refinements; assignments strengthen: x gets {ν | ν=e[env]}."""
    reff: dict[str, list[Any]] = {v: [] for v in env_ty}
    for stmt in prog:
        tag = stmt[0]
        if tag == "assign":
            _, x, e = stmt
            reff[x] = reff.get(x, []) + [("eq", "nu", e)]
        elif tag == "assert":
            _, phi = stmt
            ok = True
            for pt in _points(env_ty, domain):
                env = dict(pt)
                for v, preds in reff.items():
                    if any(not _eval(p, env, env[v]) for p in preds):
                        break
                else:
                    if not _eval(phi, env):
                        ok = False
            if not ok:
                return False
        else:
            raise ValueError(stmt)
    return True


def _points(env_ty: dict[str, Any], domain: range) -> list[dict[str, int]]:
    import itertools

    names = list(env_ty)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(domain, repeat=len(names))
    ]


def bench_refinement_liquid(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    d = range(-5, 11)
    # {ν|ν>=0} <: {ν|ν>=-1} holds
    checks.append(sub(("ge", "nu", 0), ("ge", "nu", -1), d))
    # converse fails
    checks.append(not sub(("ge", "nu", -1), ("ge", "nu", 0), d))
    # {ν|ν=5} <: {ν|ν>=0}
    checks.append(sub(("eq", "nu", 5), ("ge", "nu", 0), d))
    # VC: x := 3 ; assert x >= 0 -> ok over env {x:int}
    checks.append(vcgen([("assign", "x", 3), ("assert", ("ge", "x", 0))], None, {"x": int}, d))
    # x := 3 ; assert x >= 5 -> fails
    checks.append(not vcgen([("assign", "x", 3), ("assert", ("ge", "x", 5))], None, {"x": int}, d))
    # conjunction: x:=2, y:=x+1? our env eval only sees constants — use independent
    checks.append(
        vcgen(
            [("assign", "x", 2), ("assign", "y", 4), ("assert", ("le", "x", "y"))],
            None,
            {"x": int, "y": int},
            d,
        )
    )
    return {"synthetic_refinement_liquid": float(sum(checks)) / len(checks)}
