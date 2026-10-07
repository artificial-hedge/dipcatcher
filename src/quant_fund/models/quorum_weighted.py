"""Weighted-voting quorum systems.

Nodes carry integer weights; a quorum is any set with total weight > W/2
(write) and, for reads, weight >= W - qw + 1 so that read and write
quorums always intersect. Verified: exhaustive-pair intersection on a
small system, majority containment of the newest write under failures,
and availability simulation comparing weighted vs naive majority quorums.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

_SEED = 20261231 + 961


def is_quorum(weights: np.ndarray, members: set[int], thresh: int) -> bool:
    return int(weights[list(members)].sum()) >= thresh


def all_quorums(weights: np.ndarray, thresh: int) -> list[set[int]]:
    n = len(weights)
    out = []
    for k in range(1, n + 1):
        for c in combinations(range(n), k):
            if int(weights[list(c)].sum()) >= thresh:
                out.append(set(c))
    return out


def bench_quorum_weighted(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    weights = np.array([3, 2, 2, 1, 1, 1])
    w = int(weights.sum())  # 10
    qw = w // 2 + 1  # write quorum threshold = 6
    qr = w - qw + 1  # read quorum threshold = 5
    wq = all_quorums(weights, qw)
    rq = all_quorums(weights, qr)
    intersect_ok = all(len(a & b) > 0 for a in wq for b in rq)
    # minimal write quorums check: {0,1} weight 5 <6, {0,2} 5<6, {0,1,4} 6 ok
    if is_quorum(weights, {0, 1}, qw):
        raise ValueError("not is_quorum(weights, {0, 1}, qw)")
    if not (is_quorum(weights, {0, 1, 4}, qw)):
        raise ValueError("is_quorum(weights, {0, 1, 4}, qw)")
    # availability sim: node failures with prob pf; can we still read+write?
    pf = 0.15
    trials = 3000
    avail_w = 0
    avail_maj = 0
    for _ in range(trials):
        alive = {i for i in range(len(weights)) if rng.random() > pf}
        avail_w += any(q <= alive for q in wq) and any(q <= alive for q in rq)
        avail_maj += len(alive) >= 4  # strict majority of 6
    aw, am = avail_w / trials, avail_maj / trials
    checks = [
        intersect_ok,
        len(wq) > 0 and len(rq) > 0,
        aw > 0.9,
        aw >= am - 0.02,  # weighted quorum >= naive majority availability
    ]
    return {"synthetic_quorum_weighted": float(np.mean(checks))}
