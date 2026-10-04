"""Affine/linear use-once checking for a mini expression language.

Types carry kinds: ("lin",T) must be used exactly once, ("aff",T) at most
once, ("un",T) unrestricted. check(expr, env) walks the syntax counting
variable occurrences per context position; splitting contexts at
multiplicative pairs (like linear logic ⊗ intro splits Γ = Δ + Γ').
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1037

Expr = tuple
Env = dict[str, str]  # var -> kind


class LinearError(Exception):
    pass


def _count_uses(e: Any) -> dict[str, int]:
    if isinstance(e, str):
        return {e: 1}
    if not isinstance(e, tuple):
        return {}
    if len(e) >= 2 and e[0] == "var" and isinstance(e[1], str):
        return {e[1]: 1}
    n: dict[str, int] = {}
    for sub in e[1:]:
        for k, v in _count_uses(sub).items():
            n[k] = n.get(k, 0) + v
    return n


def check(e: Expr, env: Env) -> bool:
    """Validate kind constraints: lin vars count==1, aff<=1, un any."""
    uses = _count_uses(e)
    for v, kind in env.items():
        n = uses.get(v, 0)
        if kind == "lin" and n != 1:
            raise LinearError(f"linear {v} used {n}x")
        if kind == "aff" and n > 1:
            raise LinearError(f"affine {v} used {n}x")
    # undeclared vars treated as lin
    for v in uses:
        if v not in env and uses[v] != 1:
            raise LinearError(f"undeclared linear {v} used {uses[v]}x")
    return True


def split_ctx(env: Env, e1: Expr, e2: Expr) -> tuple[Env, Env]:
    """Context split for tensor pair (e1,e2): each linear/affine var goes
    to exactly one side; un vars go to both."""
    u1 = _count_uses(e1)
    u2 = _count_uses(e2)
    left: Env = {}
    right: Env = {}
    for v, kind in env.items():
        if kind == "un":
            left[v] = right[v] = "un"
        elif u1.get(v, 0) > 0 and u2.get(v, 0) > 0:
            raise LinearError(f"{kind} {v} used on both sides")
        elif u1.get(v, 0) > 0:
            left[v] = kind
        elif u2.get(v, 0) > 0:
            right[v] = kind
    return left, right


def bench_linear_use(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    env: Env = {"x": "lin", "y": "aff", "z": "un"}
    checks.append(check(("app", "x", "z"), env))
    # linear unused -> error
    try:
        check(("lit", 1), env)
        ok = False
    except LinearError:
        ok = True
    checks.append(ok)
    # linear used twice -> error
    try:
        check(("pair", "x", "x"), {"x": "lin"})
        ok2 = False
    except LinearError:
        ok2 = True
    checks.append(ok2)
    # affine used 0 or 1 ok, 2 bad
    checks.append(check(("lit", 1), {"y": "aff"}))
    try:
        check(("pair", "y", "y"), {"y": "aff"})
        ok3 = False
    except LinearError:
        ok3 = True
    checks.append(ok3)
    # split_ctx: un on both, lin exactly one side
    lhs, rhs = split_ctx({"x": "lin", "z": "un"}, ("var", "x"), ("var", "z"))
    checks.append(lhs == {"x": "lin", "z": "un"} and rhs == {"z": "un"})
    try:
        split_ctx({"x": "lin"}, ("var", "x"), ("var", "x"))
        ok4 = False
    except LinearError:
        ok4 = True
    checks.append(ok4)
    return {"synthetic_linear_use": float(sum(checks)) / len(checks)}
