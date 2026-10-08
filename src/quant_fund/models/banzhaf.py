"""Banzhaf power index — (1/2^{n−1}) Σ_S [v(S∪{i}) − v(S)] (SYNTHETIC)
normalized by total swings; the Penrose voting-power measure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.nucleolus import CoopGame, voting_game

FloatArray = NDArray[np.float64]


def banzhaf_index(g: CoopGame, normalized: bool = True) -> FloatArray:
    """Exact Banzhaf index by swing enumeration."""
    n = g.n
    swings = np.zeros(n)
    for s in range(2**n):
        for i in range(n):
            if not s & (1 << i) and g.v(s | (1 << i)) - g.v(s) != 0:
                swings[i] += 1.0
    raw = swings / (2.0 ** (n - 1))
    if normalized and raw.sum() > 0:
        out: FloatArray = np.asarray(raw / raw.sum())
        return out
    out2: FloatArray = np.asarray(raw)
    return out2


def bench_banzhaf(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: Banzhaf on [4,3,2,1] weights quota 6 — the big
    player's index exceeds its weight share; symmetric game uniform."""
    del seed
    out: dict[str, float] = {}
    g = voting_game(np.array([4.0, 3, 2, 1]), 6.0)
    idx = banzhaf_index(g)
    out["synthetic_banzhaf_sum"] = float(idx.sum())
    out["synthetic_banzhaf_big_share"] = float(idx[0])
    out["synthetic_banzhaf_big_gt_weight"] = float(idx[0] > 0.4)
    g2 = voting_game(np.ones(3), 2.0)
    idx2 = banzhaf_index(g2)
    out["synthetic_banzhaf_symmetric_err"] = float(np.abs(idx2 - 1.0 / 3).max())
    return out


if __name__ == "__main__":
    print(bench_banzhaf())
