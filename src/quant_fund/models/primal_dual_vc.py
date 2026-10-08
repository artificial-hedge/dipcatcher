"""Primal-dual 2-approximation for minimum vertex cover (SYNTHETIC).

Raises dual prices on edges uniformly until a vertex constraint goes
tight (sum_e y_e <= w_v saturated), freezes the tight vertex into the
cover, and continues — the classic 2-approx. Bench compares the cover
size against the brute-force minimum on the shared graph.
"""

import numpy as np

from quant_fund.models._approx_synth import G_EDGES, G_N, brute_vc


def _primal_dual() -> tuple[int, float]:
    # unit weights: y_e in [0,1], raise all active edges together
    n_edges = len(G_EDGES)
    y = np.zeros(n_edges)
    load = np.zeros(G_N)  # sum of y over incident edges
    cover: set[int] = set()
    active = np.ones(n_edges, dtype=bool)
    while active.any():
        # raise all uncovered edges until some vertex is tight
        best_t = np.inf
        tight_v = -1
        for v in range(G_N):
            if v in cover:
                continue
            inc = [e for e, (a, b) in enumerate(G_EDGES) if (a == v or b == v) and active[e]]
            if not inc:
                continue
            slack = 1.0 - load[v]
            t = slack / len(inc)
            if t < best_t:
                best_t, tight_v = t, v
        if tight_v < 0:
            break
        y[active] += best_t
        load += best_t * np.array(
            [
                sum(1 for e, (a, b) in enumerate(G_EDGES) if (a == v or b == v) and active[e])
                for v in range(G_N)
            ]
        )
        cover.add(tight_v)
        # edges incident to tight vertex are now covered
        for e, (a, b) in enumerate(G_EDGES):
            if a == tight_v or b == tight_v:
                active[e] = False
    return len(cover), float(y.sum())


def bench_primal_dual_vc(seed: int = 5203) -> dict[str, float]:
    size, dual = _primal_dual()
    truth = brute_vc(G_EDGES, G_N)
    return {
        "synthetic_pdvc_cover": float(size),
        "synthetic_pdvc_truth": float(truth),
        "synthetic_pdvc_ratio": size / truth,
        "synthetic_pdvc_dual_lb": dual,
    }
