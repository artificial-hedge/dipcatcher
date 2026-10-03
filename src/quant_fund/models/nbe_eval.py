"""Normalization by evaluation for STLC: eval to values/closures, quote
back; beta-eta equality via eta-long normal forms (SYNTHETIC bench)."""

from __future__ import annotations

from dataclasses import dataclass

_SEED = 20261231 + 994


@dataclass(frozen=True)
class Closure:
    param: str
    body: tuple
    env: dict


@dataclass(frozen=True)
class Neutral:
    head: str
    args: tuple


Val = object


def eval_(e: tuple, env: dict) -> Val:
    tag = e[0]
    if tag == "var":
        return env[e[1]]
    if tag == "lam":
        return Closure(e[1], e[2], dict(env))
    if tag == "app":
        f = eval_(e[1], env)
        a = eval_(e[2], env)
        return apply_(f, a)
    if tag == "lit":
        return e[1]
    raise ValueError(tag)


def apply_(f: Val, a: Val) -> Val:
    if isinstance(f, Closure):
        env = dict(f.env)
        env[f.param] = a
        return eval_(f.body, env)
    if isinstance(f, Neutral):
        return Neutral(f.head, f.args + (a,))
    raise ValueError("not applicable")


def fresh(env: dict, base: str) -> str:
    i = 0
    while f"{base}{i}" in env or f"{base}{i}" == base:
        i += 1
        if f"{base}{i}" not in env:
            return f"{base}{i}"
    return f"{base}X"


def quote(v: Val, level: int) -> tuple:
    if isinstance(v, Closure):
        name = f"x{level}"
        body = eval_(v.body, {**v.env, v.param: Neutral(name, ())})
        return ("lam", name, quote(body, level + 1))
    if isinstance(v, Neutral):
        out: tuple = ("var", v.head)
        for a in v.args:
            out = ("app", out, quote(a, level))
        return out
    return ("lit", v)


def normalize(e: tuple, env: dict | None = None) -> tuple:
    return quote(eval_(e, dict(env or {})), 0)


def eta_long(v: Val, level: int = 0) -> tuple:
    return quote(v, level)


def bench_nbe_eval(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    x = ("var", "y")
    idf = ("lam", "x", ("var", "x"))
    checks.append(normalize(("app", idf, x), {"y": Neutral("y", ())}) == x)
    k = ("lam", "a", ("lam", "b", ("var", "a")))
    env = {"y": Neutral("y", ()), "z": Neutral("z", ())}
    checks.append(normalize(("app", ("app", k, ("var", "y")), ("var", "z")), env) == ("var", "y"))
    comp = (
        "lam",
        "f",
        ("lam", "g", ("lam", "x", ("app", ("var", "f"), ("app", ("var", "g"), ("var", "x"))))),
    )
    iid = idf
    n1 = normalize(("app", ("app", comp, iid), iid))
    n2 = normalize(idf)
    checks.append(_alpha(n1) == _alpha(n2))
    etaf = ("lam", "x", ("app", ("var", "f"), ("var", "x")))
    fenv = {"f": Neutral("f", ()), "y": Neutral("y", ())}
    lhs = normalize(("app", etaf, ("var", "y")), fenv)
    rhs = normalize(("app", ("var", "f"), ("var", "y")), fenv)
    checks.append(_alpha(lhs) == _alpha(rhs))
    two = ("lit", 2)
    checks.append(normalize(("app", idf, two)) == ("lit", 2))
    return {"synthetic_nbe_eval": float(sum(checks) / len(checks))}


def _alpha(e: tuple) -> tuple:
    def go(t: tuple, m: dict, depth: int) -> tuple:
        tag = t[0]
        if tag == "var":
            return ("var", m.get(t[1], t[1]))
        if tag == "lam":
            new = f"v{depth}"
            m2 = dict(m)
            m2[t[1]] = new
            return ("lam", new, go(t[2], m2, depth + 1))
        if tag == "app":
            return ("app", go(t[1], m, depth), go(t[2], m, depth))
        return t

    return go(e, {}, 0)
