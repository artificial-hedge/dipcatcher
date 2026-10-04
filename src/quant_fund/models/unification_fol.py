"""First-order unification with occurs check: most general unifier (SYNTHETIC)."""

from __future__ import annotations

from typing import TypeGuard

type Term = str | tuple[Term, ...]  # atom/var or (head, arg1, ...) compound


def _is_var(t: Term) -> TypeGuard[str]:
    return isinstance(t, str) and t.startswith("?")


def _occurs(v: str, t: Term) -> bool:
    if t == v:
        return True
    return isinstance(t, tuple) and any(_occurs(v, s) for s in t)


def _subst(t: Term, s: dict[str, Term]) -> Term:
    if _is_var(t) and t in s:
        return _subst(s[t], s)
    if isinstance(t, tuple):
        return tuple(_subst(x, s) for x in t)
    return t


def unify(t1: Term, t2: Term) -> dict[str, Term] | None:
    s: dict[str, Term] = {}
    work = [(t1, t2)]
    while work:
        a, b = work.pop()
        a, b = _subst(a, s), _subst(b, s)
        if a == b:
            continue
        if _is_var(a):
            if _occurs(a, b):
                return None
            s[a] = b
        elif _is_var(b):
            if _occurs(b, a):
                return None
            s[b] = a
        elif isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b):
            work.extend(zip(a[1:], b[1:], strict=True))
        else:
            return None
    return s


def _bench_unification_fol(seed: int = 0) -> float:
    checks = []
    # f(?x, a) vs f(b, ?y) -> {?x: b, ?y: a}
    s = unify(("f", "?x", "a"), ("f", "b", "?y"))
    checks.append(s == {"?x": "b", "?y": "a"})
    # occurs check: ?x vs f(?x) fails
    checks.append(unify("?x", ("f", "?x")) is None)
    # different head symbols fail
    checks.append(unify(("f", "a"), ("g", "a")) is None)
    # nested: f(g(?x), ?y) vs f(g(h(a)), h(?z))
    s = unify(("f", ("g", "?x"), "?y"), ("f", ("g", ("h", "a")), ("h", "?z")))
    checks.append(s is not None and _subst("?x", s) == ("h", "a"))
    # constant mismatch fails
    checks.append(unify("a", "b") is None)
    # idempotent result: applying substitution equates terms
    t1, t2 = ("f", "?x", ("g", "?y")), ("f", ("h", "?z"), ("g", "c"))
    s = unify(t1, t2)
    checks.append(s is not None and _subst(t1, s) == _subst(t2, s))
    return float(sum(checks) / len(checks))


def bench_unification_fol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unification_fol": _bench_unification_fol(seed)}
