"""SYNTHETIC Selinger-style dynamic-programming join optimizer.

Left-deep DP: best[set] = cheapest plan for join set under a cost model
cost = left_card * sel + left_cost. Verify: DP matches exhaustive
permutation search on random 4-relation joins, and join commutativity.
"""

from __future__ import annotations

import itertools
import random


def _cost(plan_order: tuple[int, ...], sizes: list[float], sel: list[float]) -> float:
    card = float(sizes[plan_order[0]])
    cost = 0.0
    for j in plan_order[1:]:
        cost += card
        card *= sel[j]
    return cost


def dp_best(sizes: list[float], sel: list[float]) -> tuple[float, tuple[int, ...]]:
    n = len(sizes)
    best: dict[int, tuple[float, tuple[int, ...]]] = {}
    for i in range(n):
        best[1 << i] = (0.0, (i,))
    for mask in range(1 << n):
        if mask & (mask - 1) == 0:
            continue
        for i in range(n):
            if not mask >> i & 1:
                continue
            prev = mask ^ (1 << i)
            if prev not in best:
                continue
            c_prev, order_prev = best[prev]
            # card after order_prev = sizes[first] * prod(sel of joined)
            card = float(sizes[order_prev[0]])
            for j in order_prev[1:]:
                card *= sel[j]
            cand = (c_prev + card, order_prev + (i,))
            if mask not in best or cand[0] < best[mask][0]:
                best[mask] = cand
    return best[(1 << n) - 1]


def _exhaustive(sizes: list[float], sel: list[float]) -> float:
    return min(_cost(p, sizes, sel) for p in itertools.permutations(range(len(sizes))))


def bench_selinger_join(seed: int = 20261231 + 432) -> dict[str, float]:
    rng = random.Random(seed)
    opt = cheap = symm = 0
    trials = 60
    for _ in range(trials):
        n = rng.randrange(3, 5)
        sizes = [float(rng.randrange(10, 5000)) for _ in range(n)]
        sel = [rng.choice([0.01, 0.05, 0.2, 0.5]) for _ in range(n)]
        c_dp, order_dp = dp_best(sizes, sel)
        c_ex = _exhaustive(sizes, sel)
        opt += int(abs(c_dp - c_ex) < 1e-6 * max(1.0, c_ex))
        # DP never worse than a random order
        p = list(range(n))
        rng.shuffle(p)
        cheap += int(c_dp <= _cost(tuple(p), sizes, sel) + 1e-9)
        # symmetric: same inputs → same optimum regardless of relabel direction
        rev = list(reversed(range(n)))
        c_rev, _ = dp_best([sizes[i] for i in rev], [sel[i] for i in rev])
        symm += int(abs(c_dp - c_rev) < 1e-6 * max(1.0, c_dp))
    return {
        "synthetic_dp_is_optimal": float(opt / trials),
        "synthetic_dp_le_random": float(cheap / trials),
        "synthetic_order_invariant": float(symm / trials),
    }
