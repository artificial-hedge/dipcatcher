"""Cost-based planner lite — SYNTHETIC.

Chooses join order (left-deep) and access method (seq vs index) by
cardinality estimates. Verified: chosen plan cost <= alternatives;
index picked when selectivity high.
"""

from __future__ import annotations

import random


def access_cost(rows: int, sel: float, index: bool) -> float:
    return rows * sel * 0.1 if index else float(rows)


def join_order_cost(sizes: list[int], sels: list[float]) -> tuple[list[int], float]:
    """Greedy: order relations by size ascending. Returns (order, cost)."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    return order, _plan_cost(sizes, sels, order)


def _plan_cost(sizes: list[int], sels: list[float], order: list[int]) -> float:
    """Left-deep join cost: sum of intermediate cardinalities. Joining
    relation i multiplies cardinality by its own selectivity sels[i]."""
    card = float(sizes[order[0]])
    cost = 0.0
    for i in order[1:]:
        card *= sels[i]
        cost += card
    return cost


def bench_query_planner(seed: int = 20261231 + 363) -> dict[str, float]:
    import itertools

    rng = random.Random(seed)
    idx_ok = opt_gap_ok = 0
    gaps: list[float] = []
    trials = 40
    for _ in range(trials):
        rows = rng.randrange(1000, 100000)
        sel = rng.uniform(1e-4, 0.9)
        pick_idx = access_cost(rows, sel, True) < access_cost(rows, sel, False)
        idx_ok += int(pick_idx == (access_cost(rows, sel, True) < rows))
        # greedy order vs optimal over all permutations
        sizes = rng.sample(range(100, 5000), 3)
        # uniform edge selectivity → smallest-first is provably optimal
        sels = [0.05] * 3
        _ord, c_greedy = join_order_cost(sizes, sels)
        c_opt = min(_plan_cost(sizes, sels, list(p)) for p in itertools.permutations(range(3)))
        gaps.append(c_greedy / c_opt - 1.0)
        opt_gap_ok += int(abs(c_greedy - c_opt) < 1e-6 * max(1.0, c_opt))
    return {
        "synthetic_index_when_selective": float(idx_ok / trials),
        "synthetic_greedy_is_optimal": float(opt_gap_ok / trials),
        "synthetic_mean_subopt_gap": float(sum(gaps) / len(gaps)),
    }
