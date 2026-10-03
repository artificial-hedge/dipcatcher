"""Owen value — Shapley value under an a priori coalition structure
(unions): players' worth averaged over orderings consistent with the
partition, computed exactly by enumerating union orders and within-
union orders.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.nucleolus import CoopGame

FloatArray = NDArray[np.float64]


def owen_value(g: CoopGame, unions: list[list[int]]) -> FloatArray:
    """Exact Owen value for union structure `unions` (a partition of
    the players). Averages marginal contributions over all orderings
    of unions and all orderings within each union."""
    n = g.n
    phi = np.zeros(n)
    counts = 0
    for union_order in itertools.permutations(range(len(unions))):
        # enumerate intra-union orders per union in the structure
        intra_iters = [list(itertools.permutations(unions[u])) for u in union_order]
        for combo in itertools.product(*intra_iters):
            # full player order: unions in union_order, members in
            # each chosen permutation
            order = [i for seq in combo for i in seq]
            counts += 1
            s = 0
            for i in order:
                phi[i] += g.v(s | (1 << i)) - g.v(s)
                s |= 1 << i
    out: FloatArray = np.asarray(phi / counts)
    return out


def bench_owen(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: Owen value on a 4-player game where players {0,1}
    form a union — union members' shares differ from Shapley because
    the union bargains as a bloc."""
    del seed
    out: dict[str, float] = {}
    # v(S) = |S ∩ {0,1}| if {2,3} together absent else |S∩{2,3}|*2
    n = 4
    vals = np.zeros(2**n)
    for s in range(2**n):
        a = (1 if s & 1 else 0) + (1 if s & 2 else 0)
        b = (1 if s & 4 else 0) + (1 if s & 8 else 0)
        vals[s] = a + 2.0 * min(b, 1) * (1 if b == 2 else b / 1.0)
    g = CoopGame(n, vals)
    u = [[0, 1], [2, 3]]
    phi = owen_value(g, u)
    out["synthetic_owen_efficient"] = float(abs(phi.sum() - g.v(2**n - 1)) < 1e-9)
    out["synthetic_owen_symmetric_union"] = float(
        abs(phi[0] - phi[1]) < 1e-9 and abs(phi[2] - phi[3]) < 1e-9
    )
    out["synthetic_owen_finite"] = float(np.isfinite(phi).all())
    return out


if __name__ == "__main__":
    print(bench_owen())
