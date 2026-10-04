"""Well-founded relations, rank functions, Noetherian induction (SYNTHETIC)."""

from __future__ import annotations


def is_well_founded(rel: dict[int, list[int]], domain: list[int]) -> bool:
    """rel[x] = strict predecessors of x. Well-founded iff no infinite descending
    chain iff DAG (finite domain)."""
    color = {x: 0 for x in domain}  # 0 white,1 gray,2 black

    def dfs(x):
        color[x] = 1
        for p in rel.get(x, []):
            if color[p] == 1:
                return False
            if color[p] == 0 and not dfs(p):
                return False
        color[x] = 2
        return True

    return all(color[x] == 2 or dfs(x) for x in domain)


def rank(rel: dict[int, list[int]], x: int, memo: dict[int, int] | None = None) -> int:
    if memo is None:
        memo = {}
    if x in memo:
        return memo[x]
    preds = rel.get(x, [])
    r = 0 if not preds else 1 + max(rank(rel, p, memo) for p in preds)
    memo[x] = r
    return r


def noetherian_induct(rel: dict[int, list[int]], domain: list[int], prop) -> bool:
    """If prop(x) follows from prop on all strict predecessors for every x, then prop all."""
    memo: dict[int, int] = {}
    order = sorted(domain, key=lambda x: rank(rel, x, memo))
    for x in order:
        if all(prop(p) for p in rel.get(x, [])) and not prop(x):
            return False
    return all(prop(x) for x in domain)


def _bench_well_founded(seed: int = 0) -> float:
    checks = []
    # divisibility poset on {1..6}: x | y, x<y
    dom = [1, 2, 3, 4, 5, 6]
    rel = {x: [d for d in dom if x % d == 0 and d != x] for x in dom}
    checks.append(is_well_founded(rel, dom))
    # preds {1,2,3}: rank(1)=0, rank(2)=rank(3)=1 -> rank(6)=2
    checks.append(rank(rel, 6) == 2)
    # cyclic relation not well-founded
    checks.append(not is_well_founded({0: [1], 1: [0]}, [0, 1]))
    # prop: even rank means "good" — verify induction on subset domain
    checks.append(noetherian_induct(rel, [1, 2, 3], lambda x: x <= 3))
    return float(sum(checks) / len(checks))


def bench_well_founded(seed: int = 0) -> dict[str, float]:
    return {"synthetic_well_founded": _bench_well_founded(seed)}
