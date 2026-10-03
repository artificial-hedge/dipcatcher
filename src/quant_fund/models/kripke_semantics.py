"""Kripke semantics for modal logic: box/diamond evaluation, frame axioms (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

Model = tuple[set[int], dict[int, set[int]], dict[str, set[int]]]
Formula = Any  # ("p",name)|("not",f)|("and",f,g)|("box",f)|("dia",f)


def sat(m: Model, w: int, f: Formula) -> bool:
    worlds, rel, val = m
    tag = f[0]
    if tag == "p":
        return w in val.get(f[1], set())
    if tag == "not":
        return not sat(m, w, f[1])
    if tag == "and":
        return sat(m, w, f[1]) and sat(m, w, f[2])
    if tag == "box":
        return all(sat(m, v, f[1]) for v in rel.get(w, set()))
    if tag == "dia":
        return any(sat(m, v, f[1]) for v in rel.get(w, set()))
    raise ValueError(tag)


def valid_on_frame(worlds: set[int], rel: dict[int, set[int]], f_schema: str) -> bool:
    """Frame validity of axiom schemas: 'T' = []p->p (reflexive), '4' = []p->[][]p (transitive), 'B' = p->[]<>p (symmetric)."""
    if f_schema == "T":
        return all(w in rel.get(w, set()) for w in worlds)
    if f_schema == "4":
        return all(
            v in rel.get(w, set())
            for w in worlds
            for u in rel.get(w, set())
            for v in rel.get(u, set())
        )
    if f_schema == "B":
        return all(w in rel.get(v, set()) for w in worlds for v in rel.get(w, set()))
    raise ValueError(f_schema)


def _bench_kripke_semantics(seed: int = 0) -> float:
    checks = []
    worlds = {0, 1}
    rel = {0: {1}, 1: {1}}
    val = {"p": {1}}
    m = (worlds, rel, val)
    checks.append(sat(m, 1, ("p", "p")))
    checks.append(sat(m, 0, ("dia", ("p", "p"))))
    checks.append(
        sat(m, 0, ("box", ("dia", ("p", "p"))))
    )  # from 0 -> 1 -> 1: []p at 1 true -> []<>p at 0 true
    checks.append(sat(m, 1, ("box", ("p", "p"))))
    checks.append(sat(m, 0, ("box", ("p", "p"))))  # box only sees successors; p at 1 -> []p at 0
    checks.append(not sat(m, 0, ("p", "p")))
    checks.append(valid_on_frame(worlds, rel, "4"))
    checks.append(not valid_on_frame(worlds, rel, "T"))  # 0 not reflexive
    checks.append(not valid_on_frame(worlds, rel, "B"))  # 0->1 but 1 not ->0
    return float(sum(checks) / len(checks))


def bench_kripke_semantics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kripke_semantics": _bench_kripke_semantics(seed)}
