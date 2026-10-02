"""Association-rule mining: Apriori frequent itemsets
(Agrawal-Srikant 1994 — level-wise candidate generation
with downward-closure pruning) plus rule extraction with
confidence and lift. Synthetic bench gates recovery of a
planted co-occurrence rule."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def apriori(
    transactions: list[set[int]],
    min_support: float = 0.1,
) -> dict[frozenset[int], float]:
    """Level-wise Apriori; returns itemset → support."""
    n_t = len(transactions)
    min_cnt = max(1, int(min_support * n_t))
    items = sorted({i for t in transactions for i in t})
    sup: dict[frozenset[int], float] = {}
    prev = [frozenset([i]) for i in items]
    k = 1
    while prev:
        cand = [c for c in prev]
        counts = np.zeros(len(cand))
        for t in transactions:
            for i, c in enumerate(cand):
                if c <= t:
                    counts[i] += 1
        nxt: list[frozenset[int]] = []
        for i, c in enumerate(cand):
            if counts[i] >= min_cnt:
                sup[c] = float(counts[i] / n_t)
                nxt.append(c)
        # candidates: union of frequent k-sets sharing k−1
        prev_sets: set[frozenset[int]] = set()
        freq = list(nxt)
        for a, b in itertools.combinations(freq, 2):
            u = a | b
            if len(u) == k + 1 and all(
                frozenset(s) in {frozenset(x) for x in freq} for s in itertools.combinations(u, k)
            ):
                prev_sets.add(u)
        prev = list(prev_sets)
        k += 1
    return sup


def mine_rules(
    sup: dict[frozenset[int], float],
    min_conf: float = 0.5,
) -> list[dict[str, object]]:
    """Rules X→Y for X∪Y frequent: conf = s(X∪Y)/s(X),
    lift = conf / s(Y)."""
    rules: list[dict[str, object]] = []
    for itemset, s_xy in sup.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant in itertools.combinations(items, r):
                x = frozenset(ant)
                y = itemset - x
                s_x = sup.get(x, 0.0)
                s_y = sup.get(y, 0.0)
                if s_x <= 0 or s_y <= 0:
                    continue
                conf = s_xy / s_x
                if conf >= min_conf:
                    rules.append(
                        {
                            "ante": x,
                            "cons": y,
                            "support": s_xy,
                            "confidence": conf,
                            "lift": conf / s_y,
                        }
                    )
    return rules


def bench_association_rules(seed: int = 568) -> dict[str, float]:
    """SYNTHETIC: transactions with a planted {1,2}→{3}
    rule — mining must recover it with high confidence and
    lift > 1, and a spurious item must not surface."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    transactions: list[set[int]] = []
    for _ in range(400):
        t: set[int] = set()
        if rng.uniform() < 0.5:
            t |= {1, 2}
            if rng.uniform() < 0.9:
                t.add(3)
        else:
            if rng.uniform() < 0.15:
                t.add(3)
        for j in range(4, 9):
            if rng.uniform() < 0.08:
                t.add(j)
        if t:
            transactions.append(t)
    sup = apriori(transactions, min_support=0.05)
    rules = mine_rules(sup, min_conf=0.5)
    target = [r for r in rules if r["ante"] == frozenset({1, 2}) and r["cons"] == frozenset({3})]
    if not target:
        raise ValueError(f"rule missing: {rules}")
    best = max(target, key=lambda r: float(np.asarray(r["confidence"])))
    out["synthetic_rule_conf"] = float(np.asarray(best["confidence"]))
    out["synthetic_rule_lift"] = float(np.asarray(best["lift"]))
    if out["synthetic_rule_conf"] < 0.75:
        raise ValueError(f"rule conf off: {out['synthetic_rule_conf']}")
    if out["synthetic_rule_lift"] <= 1.0:
        raise ValueError(f"rule lift off: {out['synthetic_rule_lift']}")
    # spurious single item must not produce a strong rule into {3}
    spurious = [
        r
        for r in rules
        if r["cons"] == frozenset({3})
        and r["ante"] in (frozenset({4}), frozenset({5}))
        and float(np.asarray(r["confidence"])) > 0.75
    ]
    out["synthetic_spurious_rules"] = float(len(spurious))
    if spurious:
        raise ValueError(f"spurious rules: {spurious}")
    return out
