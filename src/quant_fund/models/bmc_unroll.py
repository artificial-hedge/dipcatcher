"""Bounded model checking by unrolling (synthetic) (SYNTHETIC).

Unrolls the transition relation to depth k on a finite-state system
and searches every trajectory for a violation of the safety
predicate. Compared against unbounded BFS reachability to measure the
depth needed for full coverage — the standard BMC picture: shallow k
misses deep counterexamples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class _Sys(NamedTuple):
    states: list[int]
    nxt: Callable[[int], list[int]]
    init: set[int]
    bad: set[int]


def _mk_counter_sys(max_x: int = 14) -> _Sys:
    """Counter: x starts 0, increments to max; bad when x == max_x-1."""
    states = list(range(max_x + 1))

    def nxt(s: int) -> list[int]:
        return [min(max_x, s + 1)]

    return _Sys(states=states, nxt=nxt, init={0}, bad={max_x - 1})


def bmc(sys: _Sys, k: int) -> dict[str, int | bool]:
    """Return violation within depth k."""
    frontier = list(sys.init)
    hit = any(s in sys.bad for s in frontier)
    for _ in range(k):
        nxt_set: set[int] = set()
        for s in frontier:
            nxt_set.update(sys.nxt(s))
        frontier = list(nxt_set)
        if any(s in sys.bad for s in frontier):
            hit = True
        if not frontier:
            break
    return {"hit": hit, "depth": k}


def _bfs_min_depth(sys: _Sys) -> int:
    """Minimum depth at which any bad state is first reached."""
    frontier = set(sys.init)
    seen = set(frontier)
    d = 0
    while frontier:
        if frontier & sys.bad:
            return d
        nxtset: set[int] = set()
        for s in frontier:
            nxtset.update(sys.nxt(s))
        nxtset -= seen
        seen |= nxtset
        frontier = nxtset
        d += 1
    return -1


def bench_bmc_unroll(seed: int = 20261231 + 223) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sys = _mk_counter_sys(14)
    d_star = _bfs_min_depth(sys)  # 13

    r_shallow = bmc(sys, k=5)
    r_exact = bmc(sys, k=d_star)
    r_deep = bmc(sys, k=d_star + 4)

    # randomized systems agree test
    agree = 0
    trials = 20
    for _ in range(trials):
        m = int(rng.integers(6, 20))
        s2 = _mk_counter_sys(m)
        d2 = _bfs_min_depth(s2)
        r2 = bmc(s2, k=d2)
        # bad reachable at exactly d2 — BMC at d2 must hit, at d2-1 must not
        r2m = bmc(s2, k=max(0, d2 - 1))
        agree += int(bool(r2["hit"]) and not bool(r2m["hit"]))
    return {
        "synthetic_min_depth": float(d_star),
        "synthetic_hit_shallow": float(bool(r_shallow["hit"])),
        "synthetic_hit_exact": float(bool(r_exact["hit"])),
        "synthetic_hit_deep": float(bool(r_deep["hit"])),
        "synthetic_agree": float(agree / trials),
    }
