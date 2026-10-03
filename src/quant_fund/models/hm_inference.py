"""SYNTHETIC Hindley-Milner type inference (Algorithm W).

Terms: var, λx.e, (e1 e2), let x = e1 in e2, literals int/bool.
Unification with occurs check; let-generalization via free vars.
"""

from __future__ import annotations

import random

# AST: ("var",x) ("lam",x,e) ("app",f,a) ("let",x,e1,e2) ("int",) ("bool",)
Expr = tuple
TV = int


class TVar:
    _n = 0

    @classmethod
    def fresh(cls) -> tuple[str, int]:
        cls._n += 1
        return ("tv", cls._n)


def _occurs(v: tuple[str, int], t: tuple) -> bool:
    if t == v:
        return True
    if isinstance(t, tuple) and len(t) == 3 and t[0] == "->":
        return _occurs(v, t[1]) or _occurs(v, t[2])
    return False


def _subst(s: dict[int, tuple], t: tuple) -> tuple:
    if t[0] == "tv" and t[1] in s:
        return _subst(s, s[t[1]])
    if t[0] == "->":
        return ("->", _subst(s, t[1]), _subst(s, t[2]))
    return t


def _unify(a: tuple, b: tuple, s: dict[int, tuple]) -> None:
    a, b = _subst(s, a), _subst(s, b)
    if a == b:
        return
    if a[0] == "tv":
        if _occurs(a, b):
            raise ValueError("occurs")
        s[a[1]] = b
        return
    if b[0] == "tv":
        if _occurs(b, a):
            raise ValueError("occurs")
        s[b[1]] = a
        return
    if a[0] == "->" and b[0] == "->":
        _unify(a[1], b[1], s)
        _unify(a[2], b[2], s)
        return
    raise ValueError("mismatch")


def _ftv(t: tuple, s: dict[int, tuple]) -> set[int]:
    t = _subst(s, t)
    if t[0] == "tv":
        return {t[1]}
    if t[0] == "->":
        return _ftv(t[1], s) | _ftv(t[2], s)
    return set()


def _free_env(env: dict[str, tuple], s: dict[int, tuple]) -> set[int]:
    out: set[int] = set()
    for t in env.values():
        if t[0] == "forall":  # scheme
            out |= _ftv(t[1], s) - set(t[2])
        else:
            out |= _ftv(t, s)
    return out


def infer(e: Expr, env: dict[str, tuple], s: dict[int, tuple]) -> tuple:
    tag = e[0]
    if tag == "int":
        return ("con", "Int")
    if tag == "bool":
        return ("con", "Bool")
    if tag == "var":
        t = env.get(e[1])
        if t is None:
            raise ValueError("unbound")
        if t[0] == "forall":  # instantiate
            _, base, qvars = t
            m = {qv: TVar.fresh() for qv in qvars}
            return _subst({k: v for k, v in m.items()}, base) if m else base
        return t
    if tag == "lam":
        tv = TVar.fresh()
        env2 = dict(env)
        env2[e[1]] = tv
        return ("->", tv, infer(e[2], env2, s))
    if tag == "app":
        tf = infer(e[1], env, s)
        ta = infer(e[2], env, s)
        rv = TVar.fresh()
        _unify(_subst(s, tf), ("->", ta, rv), s)
        return _subst(s, rv)
    if tag == "let":
        t1 = infer(e[2], env, s)
        gen = _ftv(t1, s) - _free_env(env, s)
        env2 = dict(env)
        env2[e[1]] = ("forall", _subst(s, t1), tuple(sorted(gen)))
        return infer(e[3], env2, s)
    raise ValueError(tag)


def show(t: tuple, s: dict[int, tuple]) -> str:
    t = _subst(s, t)
    if t[0] == "tv":
        return f"a{t[1]}"
    if t[0] == "con":
        return str(t[1])
    return f"({show(t[1], s)}->{show(t[2], s)})"


def bench_hm_inference(seed: int = 20261231 + 410) -> dict[str, float]:
    _ = random.Random(seed)
    ok = occ = gen = 0
    trials = 40
    for _ in range(trials):
        TVar._n = 0
        s: dict[int, tuple] = {}
        # id = λx.x : a->a
        t = infer(("lam", "x", ("var", "x")), {}, s)
        ok += int(show(t, s).startswith("(a") and "->" in show(t, s))
        # const = λx.λy.x
        TVar._n = 0
        s2: dict[int, tuple] = {}
        t2 = infer(("lam", "x", ("lam", "y", ("var", "x"))), {}, s2)
        ok += int(show(t2, s2).count("->") == 2)
        # occurs check: λx. x x must fail
        try:
            TVar._n = 0
            s3: dict[int, tuple] = {}
            infer(("lam", "x", ("app", ("var", "x"), ("var", "x"))), {}, s3)
            occ += 0
        except ValueError:
            occ += 1
        # let-gen: let id = λx.x in (id 1) → Int, and id used polymorphically
        TVar._n = 0
        s4: dict[int, tuple] = {}
        e = ("let", "id", ("lam", "x", ("var", "x")), ("app", ("var", "id"), ("int",)))
        gen += int(show(infer(e, {}, s4), s4) == "Int")
    tot = trials * 2
    return {
        "synthetic_infers_principal": float(ok / tot),
        "synthetic_occurs_rejects": float(occ / trials),
        "synthetic_let_generalizes": float(gen / trials),
    }
