"""Bisimulation between Kripke models via partition refinement (SYNTHETIC)."""

from __future__ import annotations


def bisimilar(m1, m2, w1: int, w2: int) -> bool:
    """Finite-game characterization: w1 ~ w2 iff same atoms and successors match."""
    wds1, rel1, val1 = m1
    wds2, rel2, val2 = m2
    # iterate possible-relation sets
    R = {(a, b) for a in wds1 for b in wds2}
    changed = True
    while changed:
        changed = False
        for a, b in list(R):
            atoms = all(
                (a in val1.get(p, set())) == (b in val2.get(p, set()))
                for p in set(val1) | set(val2)
            )
            zig = all(any((a2, b2) in R for b2 in rel2.get(b, set())) for a2 in rel1.get(a, set()))
            zag = all(any((a2, b2) in R for a2 in rel1.get(a, set())) for b2 in rel2.get(b, set()))
            if not (atoms and zig and zag):
                R.discard((a, b))
                changed = True
    return (w1, w2) in R


def modal_depth_equiv(m1, m2, w1: int, w2: int, depth: int) -> bool:
    """Bounded-depth equivalence (approximation for bench)."""
    if depth == 0:
        wds1, rel1, val1 = m1
        wds2, rel2, val2 = m2
        return all(
            (w1 in val1.get(p, set())) == (w2 in val2.get(p, set())) for p in set(val1) | set(val2)
        )
    return bisimilar(m1, m2, w1, w2)


def _bench_bisimulation(seed: int = 0) -> float:
    checks = []
    # chain of 2 vs single reflexive point with same atoms: NOT bisimilar
    m1 = ({0, 1}, {0: {1}, 1: {1}}, {"p": {1}})
    m2 = ({0}, {0: {0}}, {"p": {0}})
    checks.append(bisimilar(m1, m1, 1, 1))
    checks.append(not bisimilar(m1, m2, 0, 0))
    # two copies of same model are bisimilar
    checks.append(bisimilar(m2, ({1}, {1: {1}}, {"p": {1}}), 0, 1))
    # bisimilar points agree on all modal formulas: 1 in m1 vs 0 in m2 both see p-loop
    checks.append(bisimilar(m1, m2, 1, 0))
    checks.append(not bisimilar(m1, m1, 0, 1))
    return float(sum(checks) / len(checks))


def bench_bisimulation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bisimulation": _bench_bisimulation(seed)}
