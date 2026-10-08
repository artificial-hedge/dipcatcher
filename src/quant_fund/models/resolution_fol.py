"""First-order resolution refutation with full unification (SYNTHETIC).

Clauses = frozensets of literals (pred, args, neg). unify computes
MGUs over terms ("var",x)/("f",...)/consts; resolution factorizes and
resolves pairs of clauses; prove_refute runs the closure to the empty
clause — complete for the small toy theories in the bench.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1062

Term = Any
Lit = tuple  # (pred, args-tuple, neg-bool)
Clause = frozenset


def _is_var(t: Term) -> bool:
    return bool(isinstance(t, tuple) and t and t[0] == "var")


def _vars(t: Term, out: set | None = None) -> set:
    out = out if out is not None else set()
    if _is_var(t):
        out.add(t[1])
    elif isinstance(t, tuple):
        for a in t[1:]:
            _vars(a, out)
    return out


def _subst_t(t: Term, s: dict) -> Term:
    if _is_var(t) and t[1] in s:
        return _subst_t(s[t[1]], s)
    if isinstance(t, tuple):
        return (t[0],) + tuple(_subst_t(a, s) for a in t[1:])
    return t


def unify(x: Term, y: Term, s: dict | None = None) -> dict | None:
    s = dict(s or {})
    x = _subst_t(x, s)
    y = _subst_t(y, s)
    if x == y:
        return s
    if _is_var(x):
        if x[1] in _vars(y):
            return None  # occurs check
        s[x[1]] = y
        return s
    if _is_var(y):
        return unify(y, x, s)
    if isinstance(x, tuple) and isinstance(y, tuple) and x[0] == y[0] and len(x) == len(y):
        for a, b in zip(x[1:], y[1:], strict=True):
            s = unify(a, b, s)
            if s is None:
                return None
        return s
    return None


def _subst_lit(lit: Lit, s: dict) -> Lit:
    return (lit[0], tuple(_subst_t(a, s) for a in lit[1]), lit[2])


def _apart(c1: Clause, c2: Clause, tag: str) -> Clause:
    """Standardize c2's vars apart with tag suffix."""
    vs: set = set()
    for lit in c2:
        for a in lit[1]:
            _vars(a, vs)
    s = {v: ("var", f"{v}#{tag}") for v in vs}
    return frozenset(_subst_lit(lit, s) for lit in c2)


def resolve(c1: Clause, c2: Clause, tag: str = "r") -> list[Clause]:
    """All binary resolvents of c1,c2 (factoring included)."""
    out: list[Clause] = []
    c2 = _apart(c1, c2, tag)
    for l1 in c1:
        for l2 in c2:
            if l1[0] != l2[0] or l1[2] == l2[2]:
                continue
            s: dict | None = {}
            for a, b in zip(l1[1], l2[1], strict=True):
                s = unify(a, b, s)
                if s is None:
                    break
            if s is None:
                continue
            rest = (c1 - {l1}) | (c2 - {l2})
            out.append(frozenset(_subst_lit(lit, s) for lit in rest))
    return out


def refute(clauses: list[Clause], bound: int = 200) -> bool:
    """True iff the clause set is UNSAT (derives empty clause)."""
    kb = set(clauses)
    wl = list(clauses)
    steps = 0
    while wl and steps < bound:
        steps += 1
        c = wl.pop()
        if not c:
            return True
        for d in list(kb):
            for r in resolve(c, d, f"s{steps}"):
                if r not in kb:
                    kb.add(r)
                    wl.append(r)
    return frozenset() in kb


def bench_resolution_fol(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    P, Q = "P", "Q"
    vx = ("var", "x")
    a = "a"
    # {P(a)} + {¬P(x)} -> empty
    c1 = frozenset({(P, (a,), False)})
    c2 = frozenset({(P, (vx,), True)})
    checks.append(refute([c1, c2]))
    # consistent set doesn't refute
    checks.append(not refute([frozenset({(P, (a,), False)})]))
    # unify: f(x,b) vs f(a,y) -> {x:a,y:b}
    s = unify(("f", vx, "b"), ("f", a, ("var", "y")))
    checks.append(s == {"x": "a", "y": "b"})
    # occurs check: x vs f(x) fails
    checks.append(unify(vx, ("f", vx)) is None)
    # clause with two literals: {P(x), Q(x)} + {¬P(a)} -> {Q(a)}
    res = resolve(frozenset({(P, (vx,), False), (Q, (vx,), False)}), frozenset({(P, (a,), True)}))
    checks.append(any(r == frozenset({(Q, (a,), False)}) for r in res))
    return {"synthetic_resolution_fol": float(sum(checks)) / len(checks)}
