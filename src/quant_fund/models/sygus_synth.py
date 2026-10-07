"""Enumerative SyGuS for linear integer arithmetic over a stratified grammar (SYNTHETIC).

Terms: t ::= atom | t + t | t - t ; guards: g ::= t <= t ; programs:
p ::= t | ite(g, p, p). Level-bounded enumeration with
observational-equivalence pruning (one term per IO signature per level)
— the classic ESolver loop.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1018


def _eval(t: Any, env: dict[str, int]) -> int:
    tag = t[0]
    if tag == "lit":
        return int(t[1])
    if tag == "var":
        return int(env[t[1]])
    if tag == "add":
        return _eval(t[1], env) + _eval(t[2], env)
    if tag == "sub":
        return _eval(t[1], env) - _eval(t[2], env)
    if tag == "le":
        return int(_eval(t[1], env) <= _eval(t[2], env))
    if tag == "ite":
        return int(_eval(t[2], env) if _eval(t[1], env) else _eval(t[3], env))
    raise ValueError(t)


def _terms(vars_: list[str], depth: int = 1) -> list[Any]:
    """Linear terms up to `depth` operator levels."""
    atoms: list[Any] = [("lit", 0), ("lit", 1)] + [("var", v) for v in vars_]
    if depth == 0:
        return atoms
    out = list(atoms)
    for a in atoms:
        for b in atoms:
            out.append(("add", a, b))
            out.append(("sub", a, b))
    return out


def synth(spec_fn: Any, ios: list[dict[str, int]], vars_: list[str], depth: int = 2) -> Any | None:
    """Return a program consistent with ios AND spec on a bounded domain."""
    import itertools

    dom_envs = [
        dict(zip(vars_, vals, strict=True))
        for vals in itertools.product(range(-3, 4), repeat=len(vars_))
    ]

    def ok_on_spec(t: Any) -> bool:
        return all(spec_fn(_eval(t, e), e) for e in dom_envs)

    def sig(t: Any) -> tuple[int, ...]:
        return tuple(_eval(t, io) for io in ios)

    def matches(t: Any) -> bool:
        return all(_eval(t, io) == io["out"] for io in ios)

    terms = _terms(vars_, 1)
    guards = [("le", a, b) for a in terms for b in terms]
    # dedupe guards by signature
    guards = list({sig(g): g for g in guards}.values())
    terms = list({sig(t): t for t in terms}.values())

    # level 0: plain terms
    for t in terms:
        if matches(t) and ok_on_spec(t):
            return t
    # level k: ite(guard, prev_level, prev_level)
    level = terms
    for _ in range(depth):
        nxt: dict[tuple[int, ...], Any] = {}
        for g in guards:
            for a in level:
                for b in level:
                    t = ("ite", g, a, b)
                    s = sig(t)
                    if s not in nxt:
                        nxt[s] = t
                        if matches(t) and ok_on_spec(t):
                            return t
        level = list(nxt.values())
    return None


def bench_sygus_synth(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # synthesize max(x,y) from IO examples; spec = ∀ xy. f>=x ∧ f>=y ∧ (f=x|f=y)
    ios = [
        {"x": 1, "y": 2, "out": 2},
        {"x": 3, "y": 0, "out": 3},
        {"x": -1, "y": -2, "out": -1},
        {"x": 0, "y": 0, "out": 0},
    ]
    spec = lambda out, e: out >= e["x"] and out >= e["y"] and out in (e["x"], e["y"])  # noqa: E731
    t = synth(spec, ios, ["x", "y"], depth=2)
    checks.append(t is not None)
    checks.append(all(_eval(t, io) == io["out"] for io in ios) if t else False)
    checks.append(_eval(t, {"x": 5, "y": -9}) == 5 if t else False)
    # synthesize |x| — spec: f(x)>=0 ∧ f(x) in {x,-x}
    ios2 = [{"x": -2, "out": 2}, {"x": 3, "out": 3}, {"x": 0, "out": 0}]
    spec2 = lambda out, e: out >= 0 and out in (e["x"], -e["x"])  # noqa: E731
    t2 = synth(spec2, ios2, ["x"], depth=2)
    checks.append(t2 is not None and _eval(t2, {"x": -7}) == 7)
    return {"synthetic_sygus_synth": float(sum(checks)) / len(checks)}
