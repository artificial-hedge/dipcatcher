"""Three-valued (Kleene) predicate logic over logical structures — TVLA core.

Structures map individuals + predicate valuations to truth values in
{0, 1/2, 1}. Formulas evaluate three-valuedly; transitive closure is an
instrumentation predicate computed by 3-valued fixpoint (needed for
reachability in shape analysis a la Sagiv-Reps-Wilhelm).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1011

T, F, U = 1.0, 0.0, 0.5  # true, false, unknown


def _not(x: float) -> float:
    return 1.0 - x


def _and(a: float, b: float) -> float:
    return min(a, b)


def _or(a: float, b: float) -> float:
    return max(a, b)


Struct = dict[str, Any]
# {"inds": [u0,...], "preds": {"eq": {(u,u):1}, "nxt": {(u,v): t}, "sm": {u: t}}}


def eval_formula(f: Any, st: Struct, env: dict[str, int] | None = None) -> float:
    env = env or {}
    tag = f[0]
    if tag == "const":
        return float(f[1])
    if tag == "pred":
        return float(st["preds"][f[1]].get(tuple(env[v] for v in f[2:]), 0.0))
    if tag == "not":
        return _not(eval_formula(f[1], st, env))
    if tag == "and":
        return _and(eval_formula(f[1], st, env), eval_formula(f[2], st, env))
    if tag == "or":
        return _or(eval_formula(f[1], st, env), eval_formula(f[2], st, env))
    if tag == "exists":
        var = f[1]
        return max(eval_formula(f[2], st, env | {var: u}) for u in st["inds"])
    if tag == "forall":
        var = f[1]
        return min(eval_formula(f[2], st, env | {var: u}) for u in st["inds"])
    raise ValueError(f"bad formula {f}")


def tc_struct(st: Struct, base: str = "nxt") -> dict[tuple[int, int], float]:
    """3-valued transitive closure over base predicate via Kleene fixpoint."""
    inds = st["inds"]
    tc: dict[tuple[int, int], float] = dict(st["preds"][base])
    for _ in range(len(inds)):
        new: dict[tuple[int, int], float] = dict(tc)
        for i in inds:
            for j in inds:
                best = tc.get((i, j), 0.0)
                for k in inds:
                    best = max(best, _and(tc.get((i, k), 0.0), tc.get((k, j), 0.0)))
                new[(i, j)] = best
        if new == tc:
            break
        tc = new
    return tc


def bench_three_valued_logic(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # concrete list u0 -> u1 -> u2
    st: Struct = {
        "inds": [0, 1, 2],
        "preds": {
            "nxt": {(0, 1): 1.0, (1, 2): 1.0},
            "sm": {0: 0.0, 1: 0.0, 2: 0.0},
        },
    }
    f = ("exists", "v", ("pred", "nxt", "u", "v"))
    # every node except u2 has a successor -> exists v. nxt(u1,v) = 1
    checks.append(eval_formula(f, st, {"u": 1}) == 1.0)
    checks.append(eval_formula(f, st, {"u": 2}) == 0.0)
    # reachability: u0 reaches u2
    st2: Struct = dict(st)
    st2["preds"] = dict(st["preds"])
    st2["preds"]["reach"] = tc_struct(st)
    checks.append(st2["preds"]["reach"].get((0, 2)) == 1.0)
    checks.append(st2["preds"]["reach"].get((2, 0), 0.0) == 0.0)
    # summary structure: s0 -> {S} where S is summary (sm=1/2) with self-loop 1/2
    st3: Struct = {
        "inds": [0, 1],
        "preds": {
            "nxt": {(0, 1): 1.0, (1, 1): 0.5},
            "sm": {0: 0.0, 1: 0.5},
        },
    }
    st3["preds"]["reach"] = tc_struct(st3)
    # does 0 reach a node whose nxt is unknown? reach(0,1)=1, reach(1,1)=1/2
    checks.append(st3["preds"]["reach"].get((1, 1)) == 0.5)
    # kleene: U and T = U
    checks.append(eval_formula(("and", ("const", U), ("const", T)), st3) == U)
    checks.append(eval_formula(("or", ("const", U), ("const", F)), st3) == U)
    return {"synthetic_three_valued_logic": float(sum(checks)) / len(checks)}
