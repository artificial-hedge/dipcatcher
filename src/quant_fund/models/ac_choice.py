"""Axiom-of-choice witnesses and Zorn's lemma on finite posets (SYNTHETIC)."""

from __future__ import annotations

import itertools


def choice(family: list[frozenset]) -> frozenset:
    """One element from each nonempty member; exists for any finite family."""
    if any(len(s) == 0 for s in family):
        raise ValueError("no choice function: empty member")
    return frozenset(next(iter(s)) for s in family)


def is_chain(poset_rel: dict[int, set[int]], chain: set[int]) -> bool:
    for a in chain:
        for b in chain:
            if a != b and b not in poset_rel.get(a, set()) and a not in poset_rel.get(b, set()):
                return False
    return True


def upper_bounds(poset_rel: dict[int, set[int]], chain: set[int], domain: set[int]) -> set[int]:
    return {
        x
        for x in domain
        if all(
            c == x
            or c in poset_rel.get(x, set())
            or x in poset_rel.get(c, set())
            and x != c
            or x in poset_rel.get(c, set())
            for c in chain
        )
        if all(c in poset_rel.get(x, set()) | {c} for c in chain)
    }


def zorn_maximal(poset_rel: dict[int, set[int]], domain: set[int]) -> int:
    """Finite Zorn: an element with nothing strictly above it.
    Convention: poset_rel[y] = elements <= y (downward closure)."""
    for x in domain:
        ups = {y for y in domain if y != x and x in poset_rel.get(y, set())}
        if not ups:
            return x
    raise ValueError("no maximal (shouldn't happen on finite poset)")


def chains_with_ub(poset_rel: dict[int, set[int]], domain: set[int]) -> bool:
    """Every chain has an upper bound (finite poset: maximal element of chain is UB)."""
    for r in range(len(domain) + 1):
        for sub in itertools.combinations(domain, r):
            s = set(sub)
            if not is_chain(poset_rel, s):
                continue
            ub = [x for x in domain if all(x == c or c in poset_rel.get(x, set()) for c in s)]
            if not ub:
                return False
    return True


def _bench_ac_choice(seed: int = 0) -> float:
    checks = []
    fam = [frozenset({1, 2}), frozenset({"a"}), frozenset({3, 4, 5})]
    c = choice(fam)
    checks.append(len(c) == 3)
    try:
        choice([frozenset()])
        checks.append(False)
    except ValueError:
        checks.append(True)
    # poset: subsets of {0,1} under inclusion: rel[b] contains a iff a<=b
    domain = set(range(4))  # encode subsets 00,01,10,11

    def subs(x):
        return {x}

    rel = {b: {a for a in domain if (a | b) == b} for b in domain}
    checks.append(chains_with_ub(rel, domain))
    checks.append(zorn_maximal(rel, domain) == 3)  # 11 is top
    # cyclic relation 0<=1<=2<=0 is pairwise-comparable but NOT a poset:
    # transitivity fails (1 in rel[0], 2 in rel[1], but 2 notin rel[0])
    cyc = {0: {0, 1}, 1: {1, 2}, 2: {0, 2}}
    transitive = all(c in cyc[a] for a in cyc for b in cyc[a] for c in cyc[b])
    checks.append(not transitive)
    return float(sum(checks) / len(checks))


def bench_ac_choice(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ac_choice": _bench_ac_choice(seed)}
