"""de Bruijn-indexed lambda terms: shift/substitute/beta-normalization (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

Term = Any  # ("v",i) | ("lam",t) | ("app",f,x)


def lam(t: Term) -> Term:
    return ("lam", t)


def app(f: Term, x: Term) -> Term:
    return ("app", f, x)


def v(i: int) -> Term:
    return ("v", i)


def shift(t: Term, d: int, c: int = 0) -> Term:
    if t[0] == "v":
        return ("v", t[1] + d) if t[1] >= c else t
    if t[0] == "lam":
        return ("lam", shift(t[1], d, c + 1))
    return ("app", shift(t[1], d, c), shift(t[2], d, c))


def subst(t: Term, j: int, s: Term) -> Term:
    if t[0] == "v":
        if t[1] == j:
            return s
        return ("v", t[1] - 1) if t[1] > j else t
    if t[0] == "lam":
        return ("lam", subst(t[1], j + 1, shift(s, 1)))
    return ("app", subst(t[1], j, s), subst(t[2], j, s))


def norm(t: Term, fuel: int = 4000) -> Term:
    for _ in range(fuel):
        nt = step(t)
        if nt == t:
            return t
        t = nt
    raise RecursionError


def step(t: Term) -> Term:
    if t[0] == "app":
        f, x = t[1], t[2]
        if f[0] == "lam":
            return subst(f[1], 0, x)
        nf = step(f)
        if nf != f:
            return ("app", nf, x)
        return ("app", f, step(x))
    if t[0] == "lam":
        return ("lam", step(t[1]))
    return t


def _bench_de_bruijn(seed: int = 0) -> float:
    checks = []
    # (λ.0) applied to free var 0 within binder: ((λ.1) a) = 1-shifted... use closed:
    # ((λ.0) x) → x where x is Var 0 in empty env => representation: app(lam(v0), w)
    w = ("app", lam(("v", 0)), ("v", 0))
    checks.append(norm(w) == ("v", 0))
    # K = λ.λ.1 ; K a b = a. Encode a=v0,b=v0 fine (same rep)
    k = lam(lam(("v", 1)))
    checks.append(norm(app(app(k, ("v", 0)), ("v", 0))) == ("v", 0))
    # (λ.λ.0 1) a b = b 0? inner: λ.(0 applied to bound1)... use S=λ.λ.λ.(2 0)(1 0)
    s = lam(lam(lam(app(app(("v", 2), ("v", 0)), app(("v", 1), ("v", 0))))))
    skk = app(app(s, k), k)
    checks.append(norm(app(skk, ("v", 0))) == ("v", 0))
    # shift/subst: subst(λ.(0,1) , 0, x) keeps body correct
    checks.append(norm(app(lam(app(("v", 0), ("v", 1))), ("v", 0))) == app(("v", 0), ("v", 0)))
    return float(sum(checks) / len(checks))


def bench_de_bruijn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_de_bruijn": _bench_de_bruijn(seed)}
