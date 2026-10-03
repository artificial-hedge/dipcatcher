"""Euler characteristic equals alternating Betti sum (wave 280).

χ = Σ(-1)^k f_k = Σ(-1)^k b_k for any finite complex — verify on random flag
complexes plus the named oracles.
"""

import numpy as np

from quant_fund.models.simp_betti import betti

_SEED = 20261231 + 766


def _flag(rng: np.random.RandomState, n: int, p: float) -> list[list[tuple[int, ...]]]:
    adj = rng.rand(n, n) < p
    np.fill_diagonal(adj, False)
    edges: list[tuple[int, ...]] = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    tris: list[tuple[int, ...]] = [
        (i, j, k) for (i, j) in edges for k in range(j + 1, n) if adj[i, k] and adj[j, k]
    ]
    tets: list[tuple[int, ...]] = [
        (i, j, k, l_)
        for (i, j, k) in tris
        for l_ in range(k + 1, n)
        if adj[i, l_] and adj[j, l_] and adj[k, l_]
    ]
    out: list[list[tuple[int, ...]]] = [[(i,) for i in range(n)], edges, tris, tets]
    return out


def bench_euler_char(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(6):
        comp = _flag(rng, 7, 0.45)
        chi_f = sum((-1) ** k * len(s) for k, s in enumerate(comp))
        b = betti(comp)
        chi_b = sum((-1) ** k * bk for k, bk in enumerate(b))
        ok += int(chi_f == chi_b)
    return {"synthetic_euler_betti": float(ok == 6)}
