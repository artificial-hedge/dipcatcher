"""Pattern-fragment unification with metavariables (Miller): flex-rigid,
occurs check, distinct-bound-var spines (SYNTHETIC bench only)."""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 996

Term = Any


def is_meta(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) > 0 and t[0] == "?"


def spine_vars(t: tuple) -> list[str]:
    args = t[2]
    vars_: list[str] = []
    for a in args:
        if not (isinstance(a, tuple) and a[0] == "var"):
            raise ValueError("non-pattern spine")
        vars_.append(str(a[1]))
    if len(set(vars_)) != len(vars_):
        raise ValueError("non-distinct spine")
    return vars_


def occurs(name: str, t: Term) -> bool:
    if is_meta(t):
        if t[1] == name:
            return True
        return any(occurs(name, a) for a in t[2])
    if isinstance(t, tuple):
        return any(occurs(name, x) for x in t[1:])
    return False


def deref(t: Term, subst: dict) -> Term:
    while is_meta(t) and t[1] in subst:
        lam_vars, body = subst[t[1]]
        args = [deref(a, subst) for a in t[2]]
        t = body
        for v, a in zip(lam_vars, args, strict=True):
            t = _subst(t, v, a)
    return t


def _subst(t: Term, var: str, arg: Term) -> Term:
    if isinstance(t, tuple):
        if t[0] == "var" and t[1] == var:
            return arg
        if t[0] == "lam" and t[1] == var:
            return t
        if is_meta(t):
            return ("?", t[1], tuple(_subst(a, var, arg) for a in t[2]))
        return tuple(_subst(x, var, arg) for x in t)
    return t


def unify(t1: Term, t2: Term, subst: dict) -> dict:
    t1 = deref(t1, subst)
    t2 = deref(t2, subst)
    if is_meta(t1):
        return _bind(t1, t2, subst)
    if is_meta(t2):
        return _bind(t2, t1, subst)
    if isinstance(t1, tuple) and isinstance(t2, tuple):
        if t1[0] == "lam" and t2[0] == "lam":
            s2 = _subst(t2[2], t2[1], ("var", t1[1]))
            return unify(t1[2], s2, subst)
        if len(t1) != len(t2):
            raise ValueError("arity mismatch")
        for a, b in zip(t1, t2, strict=True):
            subst = unify(a, b, subst)
        return subst
    if t1 == t2:
        return subst
    raise ValueError(f"rigid mismatch {t1} {t2}")


def _bind(m: tuple, t: Term, subst: dict) -> dict:
    name = m[1]
    vars_ = spine_vars(m)
    if occurs(name, t):
        raise ValueError("occurs check")
    # prune: t may reference spine vars only via var nodes (pattern fragment
    # guarantees abstraction is legal since spine vars are bound outside)
    subst[name] = (vars_, t)
    return subst


def solve(t1: Term, t2: Term) -> dict:
    return unify(t1, t2, {})


def bench_unify_meta(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    s = solve(("?", "m", (("var", "x"),)), ("app", "f", ("var", "x"), ("var", "x")))
    checks.append(s["m"] == (["x"], ("app", "f", ("var", "x"), ("var", "x"))))
    try:
        solve(("?", "m", (("var", "x"),)), ("app", "g", ("?", "m", (("var", "x"),))))
        checks.append(False)
    except ValueError:
        checks.append(True)
    s2 = solve(("?", "m", (("var", "x"),)), ("var", "x"))
    checks.append(s2["m"] == (["x"], ("var", "x")))
    s3 = solve(
        ("pair", ("?", "a", ()), ("?", "b", ())),
        ("pair", ("lit", 1), ("lit", 2)),
    )
    checks.append(s3["a"][1] == ("lit", 1) and s3["b"][1] == ("lit", 2))
    try:
        solve(("lit", 1), ("lit", 2))
        checks.append(False)
    except ValueError:
        checks.append(True)
    try:
        solve(("var", "x"), ("var", "y"))
        checks.append(False)
    except ValueError:
        checks.append(True)
    # flex-flex
    s4 = solve(("?", "p", (("var", "x"),)), ("?", "q", (("var", "x"),)))
    checks.append(len(s4) == 1)
    # resolved meta is reused on later equations
    s5 = unify(
        ("pair", ("?", "m", ()), ("?", "m", ())),
        ("pair", ("lit", 9), ("lit", 9)),
        {},
    )
    checks.append(s5["m"][1] == ("lit", 9))
    return {"synthetic_unify_meta": float(sum(checks) / len(checks))}
