"""Robinson first-order unification with occurs check (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

Term = Any  # ("var",name) | ("fn",f,[args])
Subst = dict[str, Term]


def walk(t: Term, s: Subst) -> Term:
    while isinstance(t, tuple) and t[0] == "var" and t[1] in s:
        t = s[t[1]]
    return t


def occurs(v: str, t: Term, s: Subst) -> bool:
    t = walk(t, s)
    if t[0] == "var":
        return bool(t[1] == v)
    return bool(any(occurs(v, a, s) for a in t[2]))


def unify(x: Term, y: Term, s: Subst | None = None) -> Subst | None:
    if s is None:
        s = {}
    x, y = walk(x, s), walk(y, s)
    if x == y:
        return s
    if x[0] == "var":
        if occurs(x[1], y, s):
            return None
        return {**s, x[1]: y}
    if y[0] == "var":
        if occurs(y[1], x, s):
            return None
        return {**s, y[1]: x}
    if x[0] != "fn" or y[0] != "fn" or x[1] != y[1] or len(x[2]) != len(y[2]):
        return None
    for ax, ay in zip(x[2], y[2], strict=True):
        s = unify(ax, ay, s)
        if s is None:
            return None
    return s


def apply_subst(t: Term, s: Subst) -> Term:
    t = walk(t, s)
    if t[0] == "var":
        return t
    return ("fn", t[1], [apply_subst(a, s) for a in t[2]])


def _bench_unification(seed: int = 0) -> float:
    checks = []
    s = unify(("var", "x"), ("fn", "f", [("var", "y")]))
    checks.append(s is not None and s["x"][0] == "fn")
    # occurs check: x vs f(x) fails
    checks.append(unify(("var", "x"), ("fn", "f", [("var", "x")])) is None)
    # f(g(x),y) vs f(g(a),b)
    s2 = unify(
        ("fn", "f", [("fn", "g", [("var", "x")]), ("var", "y")]),
        ("fn", "f", [("fn", "g", [("fn", "a", [])]), ("fn", "b", [])]),
    )
    checks.append(
        s2 is not None
        and apply_subst(("var", "x"), s2) == ("fn", "a", [])
        and apply_subst(("var", "y"), s2) == ("fn", "b", [])
    )
    # clash: f vs g
    checks.append(unify(("fn", "f", []), ("fn", "g", [])) is None)
    # shared var consistency: f(x,x) vs f(a,b) fails
    checks.append(
        unify(
            ("fn", "f", [("var", "x"), ("var", "x")]),
            ("fn", "f", [("fn", "a", []), ("fn", "b", [])]),
        )
        is None
    )
    return float(sum(checks) / len(checks))


def bench_unification(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unification": _bench_unification(seed)}
