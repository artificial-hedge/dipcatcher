"""Dense sets and generic filters on finite posets (SYNTHETIC)."""

from __future__ import annotations

Cond = dict[int, int]


def dense_below(conds: list[Cond], target_len: int) -> list[Cond]:
    """The dense set D_k = {p : |dom(p)| >= k} for Cohen forcing:
    every condition extends to one with at least k points."""
    return [c for c in conds if len(c) >= target_len]


def is_dense(d: list[Cond], all_conds: list[Cond]) -> bool:
    """D is dense iff every condition has an extension inside D."""

    def extends(p: Cond, q: Cond) -> bool:
        return all(k in p and p[k] == v for k, v in q.items())

    return all(any(extends(dc, c) for dc in d) for c in all_conds)


def filter_meets_dense(g: list[Cond], d: list[Cond]) -> bool:
    """Generic G meets every dense set."""
    return any(gi in d or gi == di for gi in g for di in d)


def _bench_dense_filter(seed: int = 0) -> float:
    checks = []
    all_conds = [
        {},
        {0: 0},
        {0: 1},
        {0: 0, 1: 0},
        {0: 0, 1: 1},
        {0: 1, 1: 0},
        {0: 1, 1: 1},
    ]
    d2 = dense_below(all_conds, 2)
    checks.append(len(d2) == 4)
    checks.append(is_dense(d2, all_conds))
    # generic = branch filter { {} , {0:1}, {0:1,1:0} } meets D_2
    g = [{}, {0: 1}, {0: 1, 1: 0}]
    checks.append(filter_meets_dense(g, d2))
    # the filter { {} } alone misses D_2 -> not generic for n=2
    checks.append(not filter_meets_dense([{}], d2))
    # upward closure: filter contains supersets -> {0:1,1:0} implies {0:1}?
    # a filter is UPWARD closed: if p in G and p <= q then q in G;
    # here {0:1,1:0} <= {0:1} so {0:1} must be in G -> it is
    checks.append({0: 1} in g)
    # incompatibility-free: all pairs in g compatible
    checks.append(all(all(b[k] == a[k] for k in a if k in b) for a in g for b in g))
    return float(sum(checks) / len(checks))


def bench_dense_filter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dense_filter": _bench_dense_filter(seed)}
