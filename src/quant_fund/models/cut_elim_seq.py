"""Gentzen cut elimination on atomic sequents (SYNTHETIC)."""

from __future__ import annotations


def cut_free_provable(facts: list[tuple[int, int]], a: int, c: int) -> bool:
    """Graph reachability = cut-free derivability of a |- c from
    atomic rules x |- y."""
    seen = {a}
    frontier = [a]
    while frontier:
        u = frontier.pop()
        for x, y in facts:
            if x == u and y not in seen:
                seen.add(y)
                frontier.append(y)
    return c in seen


def _bench_cut_elim_seq(seed: int = 0) -> float:
    checks = []
    facts = [(0, 1), (1, 2), (2, 3)]
    # chain a|-b, b|-c gives a|-c via cut, also cut-free reachable
    checks.append(cut_free_provable(facts, 0, 3))
    # no path -> unprovable
    checks.append(not cut_free_provable(facts, 3, 0))
    # reflexive sequents a|-a always derivable (axiom)
    checks.append(cut_free_provable(facts, 1, 1))
    # disconnected node
    checks.append(not cut_free_provable(facts + [(5, 6)], 0, 6))
    # transitivity chain length 3
    checks.append(cut_free_provable(facts, 0, 2))
    return float(sum(checks) / len(checks))


def bench_cut_elim_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cut_elim_seq": _bench_cut_elim_seq(seed)}
