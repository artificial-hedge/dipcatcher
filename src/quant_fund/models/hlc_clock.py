"""Hybrid Logical Clock (HLC): Lamport-style causality over physical time.

hlc = (l, c): l tracks max observed physical time, c disambiguates events
at equal l. Verified: happens-before edges imply strict HLC order, l stays
within a bounded drift of physical time, and l never falls behind pt.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 956

Hlc = tuple[int, int]


def hlc_local(pt: int, cur: Hlc) -> Hlc:
    lt, c = cur
    if pt > lt:
        return (pt, 0)
    return (lt, c + 1)


def hlc_recv(pt: int, cur: Hlc, msg: Hlc) -> Hlc:
    lt, c = cur
    lm, cm = msg
    if pt > lt and pt > lm:
        return (pt, 0)
    if lt == lm:
        return (lt, max(c, cm) + 1)
    if lt > lm:
        return (lt, c + 1)
    return (lm, cm + 1)


def _lt(a: Hlc, b: Hlc) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


def bench_hlc_clock(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_proc, n_ev = 4, 120
    stamps: list[Hlc] = []
    pts: list[int] = []
    causal_before: list[tuple[int, int]] = []
    last: dict[int, int] = {}  # proc -> last event idx
    sends = 0
    cur_hlc: dict[int, Hlc] = {p: (0, 0) for p in range(n_proc)}
    cur_pt: dict[int, int] = {p: 0 for p in range(n_proc)}
    for i in range(n_ev):
        p = int(rng.integers(n_proc))
        pt = int(cur_pt[p] + rng.integers(1, 5))
        src = int(rng.integers(i)) if i > 4 and rng.random() < 0.2 else None
        h = hlc_local(pt, cur_hlc[p])
        if src is not None:
            h = hlc_recv(pt, cur_hlc[p], stamps[src])
        stamps.append(h)
        pts.append(pt)
        if p in last:
            causal_before.append((last[p], i))
        if src is not None:
            causal_before.append((src, i))
            sends += 1
        last[p] = i
        cur_hlc[p] = h
        cur_pt[p] = pt
    checks = [
        all(_lt(stamps[a], stamps[b]) for a, b in causal_before),
        all(stamps[i][0] - pts[i] <= 40 for i in range(n_ev)),
        all(stamps[i][0] >= pts[i] for i in range(n_ev)),  # l never behind pt
        len(causal_before) == n_ev - n_proc + sends and sends > 0,
    ]
    return {"synthetic_hlc_clock": float(np.mean(checks))}
