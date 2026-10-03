"""Fault-tree analysis: gate-level probability + minimal cut sets.

Top-down expansion of the tree yields cut sets (AND = union of children
cuts, OR = union of alternatives); subsumed sets are pruned to the
minimal cut sets. Gate evaluation gives the exact top-event probability
for trees without repeated basic events, checked against a Monte-Carlo
simulation of the same tree.
"""

import numpy as np

from quant_fund.models._rel_synth import TREE_P, fault_tree, fault_tree_top_prob


def _cut_sets(node) -> list[frozenset[str]]:
    if isinstance(node, str):
        return [frozenset({node})]
    gate = node["gate"]
    child_sets = [_cut_sets(c) for c in node["children"]]
    if gate == "or":
        return [cs for group in child_sets for cs in group]
    # AND: cartesian union of one cut from each child
    out: list[frozenset[str]] = [frozenset()]
    for group in child_sets:
        out = [a | b for a in out for b in group]
    return out


def _minimal(sets: list[frozenset[str]]) -> list[frozenset[str]]:
    keep = []
    for cs in sets:
        if not any((other < cs) for other in sets):
            keep.append(cs)
    return keep


def _eval_prob(node, p: dict[str, float]) -> float:
    if isinstance(node, str):
        return p[node]
    probs = [_eval_prob(c, p) for c in node["children"]]
    if node["gate"] == "or":
        return float(1.0 - np.prod([1.0 - q for q in probs]))
    return float(np.prod(probs))


def _mc_prob(node, p: dict[str, float], n: int = 60000, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    vals = {k: rng.uniform(size=n) < v for k, v in p.items()}

    def ev(nd):
        if isinstance(nd, str):
            return vals[nd]
        arr = np.stack([ev(c) for c in nd["children"]])
        return arr.any(axis=0) if nd["gate"] == "or" else arr.all(axis=0)

    return float(np.mean(ev(node)))


def bench_fault_tree(seed: int = 4903) -> dict[str, float]:
    tree = fault_tree()
    mcs = _minimal(_cut_sets(tree))
    top = _eval_prob(tree, TREE_P)
    mc = _mc_prob(tree, TREE_P, seed=seed)
    return {
        "synthetic_ft_top_prob": top,
        "synthetic_ft_top_true": fault_tree_top_prob(),
        "synthetic_ft_top_err": abs(top - fault_tree_top_prob()),
        "synthetic_ft_mc_err": abs(top - mc),
        "synthetic_ft_n_min_cut": float(len(mcs)),
        "synthetic_ft_min_cut_size": float(min(len(cs) for cs in mcs)),
    }
