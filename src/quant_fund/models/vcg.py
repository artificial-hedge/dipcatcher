"""VCG mechanism — welfare-maximizing allocation with Clarke pivot
payments. Covers single-item second-price auction (special case) and
general combinatorial allocations via exhaustive welfare search.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def vcg_auction(bids: FloatArray, n_items: int = 1) -> tuple[FloatArray, FloatArray]:
    """Single-parameter VCG: allocate each of `n_items` identical items
    to the highest bids; charge Clarke pivot payments.

    bids: per-bidder per-unit valuations (flat bids). Returns
    (allocation_vector, payments).
    """
    order = np.argsort(-bids)
    alloc = np.zeros(len(bids))
    alloc[order[:n_items]] = 1.0
    pay = np.zeros(len(bids))
    for i in np.where(alloc > 0)[0]:
        # pivotal externality: the bid that would have won without i
        others = np.delete(np.sort(bids)[::-1], np.where(np.sort(bids)[::-1] == bids[i])[0][0])
        pay[i] = others[n_items - 1] if len(others) >= n_items else 0.0
    return alloc, pay


def vcg_combinatorial(
    bids: list[tuple[tuple[int, ...], float]],
    n_agents: int,
    n_items: int,
) -> tuple[list[tuple[int, ...]], FloatArray]:
    """Combinatorial VCG: each agent bids on bundles.

    bids: list of (bundle, value) pairs, one per agent (single-minded).
    Returns (winning bundles, payments).
    """
    bundles = [b for b, _ in bids]
    vals = np.array([v for _, v in bids])

    def welfare(chosen: list[int]) -> float:
        used: set[int] = set()
        tot = 0.0
        for i in chosen:
            if any(it in used for it in bundles[i]):
                return -1e18
            used.update(bundles[i])
            tot += vals[i]
        return tot

    idx = list(range(len(bids)))
    best, best_set = -1e18, []
    for r in range(len(idx) + 1):
        for combo in itertools.combinations(idx, r):
            w = welfare(list(combo))
            if w > best:
                best, best_set = w, list(combo)
    winners = set(best_set)
    pay = np.zeros(n_agents)
    for i in range(n_agents):
        if i not in winners:
            continue
        # welfare of others with i present vs. optimal without i
        others = [j for j in best_set if j != i]
        w_with = sum(vals[j] for j in others)
        rem = [j for j in idx if j != i]
        best_wo = -1e18
        for r in range(len(rem) + 1):
            for combo in itertools.combinations(rem, r):
                w = welfare(list(combo))
                best_wo = max(best_wo, w)
        pay[i] = best_wo - w_with
    return [bundles[i] for i in best_set], pay


def bench_vcg(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: second-price special case (winner pays 2nd bid) and a
    combinatorial instance where bundling beats disjoint winners."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    bids = np.array([10.0, 7.0, 4.0, 3.0])
    alloc, pay = vcg_auction(bids, 1)
    out["synthetic_vcg_winner_correct"] = float(alloc[0] == 1)
    out["synthetic_vcg_second_price"] = float(abs(pay[0] - 7.0) < 1e-9)
    # combinatorial: A bids {0,1}=12 > B{0}=6 + C{1}=5 (=11)
    bundles, pay2 = vcg_combinatorial([((0, 1), 12.0), ((0,), 6.0), ((1,), 5.0)], 3, 2)
    out["synthetic_vcg_combo_winner"] = float(bundles == [(0, 1)])
    # A's Clarke pivot: without A, B+C take welfare 11; others get 0
    out["synthetic_vcg_combo_pay"] = float(pay2[0])
    out["synthetic_vcg_combo_pay_ok"] = float(abs(pay2[0] - 11.0) < 1e-9)
    # dominant-strategy incentive: truthful bidding is optimal
    vals = rng.uniform(1, 10, 20)
    for i in range(20):
        others = np.delete(vals, i)
        a_t, p_t = vcg_auction(np.insert(others, i, vals[i]), 1)
        u_truth = (vals[i] - p_t[i]) * a_t[i]
        best_u = u_truth
        for overbid in np.linspace(0.5, 1.5, 5) * vals[i]:
            a_l, p_l = vcg_auction(np.insert(others, i, overbid), 1)
            best_u = max(best_u, (vals[i] - p_l[i]) * a_l[i])
        if not (best_u <= u_truth + 1e-9 or abs(best_u - u_truth) < 1e-6):
            raise ValueError("best_u <= u_truth + 1e-9 or abs(best_u - u_truth) < 1e-6")
    out["synthetic_vcg_dsic"] = 1.0
    return out


if __name__ == "__main__":
    print(bench_vcg())
