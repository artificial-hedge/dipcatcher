"""Fiduccia-Mattheyses hypergraph bipartitioning (one balanced pass) (SYNTHETIC).

Cells with unit area, hyperedges; a single FM pass moves the max-gain cell
under the balance constraint, tracks the best prefix of the move sequence,
then rolls back. Verified against a planted bisection: the pass must
strictly reduce the initial cut and recover a near-planted cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SEED = 20261231 + 950


@dataclass
class HyperGraph:
    n_cells: int
    edges: list[list[int]]


def _cut(hg: HyperGraph, part: np.ndarray) -> int:
    c = 0
    for e in hg.edges:
        s = {part[i] for i in e}
        if len(s) > 1:
            c += 1
    return c


def _gains(hg: HyperGraph, part: np.ndarray, locked: np.ndarray) -> np.ndarray:
    g = np.zeros(hg.n_cells, dtype=int)
    for e in hg.edges:
        cnt = [0, 0]
        for i in e:
            cnt[part[i]] += 1
        for i in e:
            side = part[i]
            if cnt[side] == 1:
                g[i] += 1  # sole cell on its side: moving removes the cut
            elif cnt[1 - side] == 0:
                g[i] -= 1  # all pins on same side: moving adds the cut
    return g


def fm_pass(hg: HyperGraph, part: np.ndarray, tol: float = 0.15) -> np.ndarray:
    part = part.copy()
    n = hg.n_cells
    locked = np.zeros(n, dtype=bool)
    g = _gains(hg, part, locked)
    sizes = [int((part == 0).sum()), int((part == 1).sum())]
    lo = int(n * (0.5 - tol))
    hi = int(n * (0.5 + tol))
    moves: list[int] = []
    best_cut = _cut(hg, part)
    best_k = 0
    for k in range(n):
        cand = np.where(~locked)[0]
        bal_ok = [i for i in cand if sizes[part[i]] - 1 >= lo and sizes[1 - part[i]] + 1 <= hi]
        if not bal_ok:
            break
        i = int(max(bal_ok, key=lambda x: g[x]))
        locked[i] = True
        sizes[part[i]] -= 1
        part[i] ^= 1
        sizes[part[i]] += 1
        g = _gains(hg, part, locked)
        moves.append(i)
        c = _cut(hg, part)
        if c < best_cut:
            best_cut = c
            best_k = k + 1
    # roll back to best prefix
    for k in range(best_k, len(moves)):
        part[moves[k]] ^= 1
    return part


def bench_fm_partition(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = 40
    # planted bisection: 8 dense intra-cluster hyperedges + 4 bridging
    true_part = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.uint8)
    edges: list[list[int]] = []
    for _ in range(28):
        side = int(rng.random() < 0.5)
        pool = np.where(true_part == side)[0]
        edges.append(sorted(rng.choice(pool, size=3, replace=False).tolist()))
    for _ in range(4):
        edges.append(sorted(rng.choice(n, size=3, replace=False).tolist()))
    hg = HyperGraph(n, edges)
    planted = _cut(hg, true_part)
    # random start
    start = rng.permutation(true_part)
    c0 = _cut(hg, start)
    out = fm_pass(hg, start, tol=0.2)
    c1 = _cut(hg, out)
    checks = [c1 <= planted + 2, c1 < c0, _cut(hg, fm_pass(hg, out, 0.2)) <= c1 + 1]
    return {"synthetic_fm_partition": float(np.mean(checks))}
