"""Free term algebra on the w337 unifier's term form (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.unification import apply_subst, unify

Term = tuple  # ("var",name) | ("fn",name,[args])


def tvar(name: str) -> Term:
    return ("var", name)


def tfun(name: str, *args: Term) -> Term:
    return ("fn", name, list(args))


def size(t: Term) -> int:
    if t[0] == "var":
        return 1
    return 1 + sum(size(a) for a in t[2])


def depth(t: Term) -> int:
    if t[0] == "var":
        return 0
    return 1 + max((depth(a) for a in t[2]), default=0)


def _bench_term_algebra(seed: int = 0) -> float:
    checks = []
    x = tvar("x")
    t = tfun("f", x, tfun("g", tvar("y")))
    checks.append(size(t) == 4)
    checks.append(depth(t) == 2)
    # unify f(x, g(y)) with f(a, g(b)) where a=g(z),b=g(w)
    u = unify(t, tfun("f", tfun("g", tvar("z")), tfun("g", tvar("w"))))
    checks.append(u is not None)
    checks.append(u is not None and u.get("x") == ("fn", "g", [("var", "z")]))
    # occurs check: x = f(x) fails
    checks.append(unify(x, tfun("f", x)) is None)
    # substitution application
    checks.append(apply_subst(x, {"x": ("fn", "h", [])}) == ("fn", "h", []))
    return float(sum(checks) / len(checks))


def bench_term_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_term_algebra": _bench_term_algebra(seed)}
