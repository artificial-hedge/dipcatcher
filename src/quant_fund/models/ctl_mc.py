"""CTL model checking over finite Kripke structures.

State space = int states; labels per state; transition relation. Fixpoint
semantics: EX p = pre(p); EF p = μZ. p∨pre(Z); EG p = νZ. p∧pre(Z);
EU p q = μZ. q∨(p∧pre(Z)); AF p = μZ. p∧post-all(Z); AG p = νZ. p∧post-all(Z).
"""

from __future__ import annotations

_SEED = 20261231 + 1044

Kripke = tuple[set[int], dict[int, list[int]], dict[int, set[str]]]


def _pre(succ: dict[int, list[int]], states: set[int], z: set[int]) -> set[int]:
    return {s for s in states if any(t in z for t in succ.get(s, []))}


def _pre_all(succ: dict[int, list[int]], states: set[int], z: set[int]) -> set[int]:
    return {s for s in states if succ.get(s, []) and all(t in z for t in succ[s])}


def sat(formula: tuple, k: Kripke) -> set[int]:
    states, succ, lab = k
    tag = formula[0]
    if tag == "atom":
        return {s for s in states if formula[1] in lab.get(s, set())}
    if tag == "not":
        return states - sat(formula[1], k)
    if tag == "and":
        return sat(formula[1], k) & sat(formula[2], k)
    if tag == "or":
        return sat(formula[1], k) | sat(formula[2], k)
    if tag == "ex":
        return _pre(succ, states, sat(formula[1], k))
    if tag == "ef":
        p = sat(formula[1], k)
        z = set(p)
        while True:
            z2 = z | _pre(succ, states, z)
            if z2 == z:
                return z
            z = z2
    if tag == "eg":
        p = sat(formula[1], k)
        z = set(states) & set(p)
        while True:
            z2 = z & _pre(succ, states, z)
            if z2 == z:
                return z
            z = z2
    if tag == "eu":
        p = sat(formula[1], k)
        q = sat(formula[2], k)
        z = set(q)
        while True:
            z2 = z | (p & _pre(succ, states, z))
            if z2 == z:
                return z
            z = z2
    if tag == "af":
        p = sat(formula[1], k)
        z = set(p)
        while True:
            z2 = z | _pre_all(succ, states, z)
            if z2 == z:
                return z
            z = z2
    if tag == "ag":
        p = sat(formula[1], k)
        z = set(states) & set(p)
        while True:
            z2 = z & _pre_all(succ, states, z)
            if z2 == z:
                return z
            z = z2
    raise ValueError(formula)


def holds(formula: tuple, k: Kripke, init: int) -> bool:
    return init in sat(formula, k)


def bench_ctl_mc(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # s0->s1->s2 ; labels: s0:{}, s1:{p}, s2:{p,q}
    k: Kripke = ({0, 1, 2}, {0: [1], 1: [2], 2: []}, {0: set(), 1: {"p"}, 2: {"p", "q"}})
    checks.append(holds(("ex", ("atom", "p")), k, 0))
    checks.append(holds(("ef", ("atom", "q")), k, 0))
    checks.append(not holds(("ag", ("atom", "p")), k, 0))
    checks.append(holds(("eu", ("atom", "p"), ("atom", "q")), k, 1))
    # deadlock-free cycle: s0<->s1, p on s1: EG p false at s0; EF p true
    k2: Kripke = ({0, 1}, {0: [1], 1: [0]}, {0: set(), 1: {"p"}})
    checks.append(holds(("ef", ("atom", "p")), k2, 0))
    checks.append(not holds(("ag", ("atom", "p")), k2, 0))
    return {"synthetic_ctl_mc": float(sum(checks)) / len(checks)}
