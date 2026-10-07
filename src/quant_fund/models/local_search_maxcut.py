"""Local-search max-cut — the 1/2-approximation (SYNTHETIC).

Greedy local improvement: move any vertex whose swap increases the
cut until no improving move exists (local optimum guarantees >= 1/2
of edges cut). Bench compares against the brute-force max cut on the
shared 10-node graph and a random-partition baseline.
"""

import numpy as np

from quant_fund.models._approx_synth import G_EDGES, G_N, brute_maxcut


def _local_search() -> tuple[int, int]:
    side = np.zeros(G_N, dtype=bool)
    # seed deterministically: put evens in S
    side[::2] = True
    moves = 0

    def delta(v: int) -> int:
        d = 0
        for a, b in G_EDGES:
            if a == v or b == v:
                other = b if a == v else a
                d += -1 if side[other] != side[v] else 1
        return d

    improved = True
    while improved:
        improved = False
        for v in range(G_N):
            if delta(v) > 0:
                side[v] = not side[v]
                moves += 1
                improved = True
    cut = sum(side[u] != side[v] for u, v in G_EDGES)
    return int(cut), moves


def bench_local_search_maxcut(seed: int = 5209) -> dict[str, float]:
    cut, moves = _local_search()
    truth = brute_maxcut(G_EDGES, G_N)
    rng = np.random.default_rng(seed)
    rand = max(
        sum(1 for u, v in G_EDGES if assign[u] != assign[v])
        for assign in (rng.integers(0, 2, size=G_N).astype(bool) for _ in range(200))
    )
    return {
        "synthetic_lsmc_cut": float(cut),
        "synthetic_lsmc_truth": float(truth),
        "synthetic_lsmc_ratio": cut / truth,
        "synthetic_lsmc_moves": float(moves),
        "synthetic_lsmc_random_best": float(rand),
    }
