"""SYNTHETIC deadlock detection + Banker's safety.

Wait-for-graph cycle detection via DFS; Banker's algorithm verified
against exhaustive ordering enumeration of the safety condition.
"""

from __future__ import annotations

import itertools
import random


def has_deadlock(wait_for: dict[int, set[int]]) -> bool:
    color: dict[int, int] = {}  # 0=white,1=gray,2=black

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in wait_for.get(u, set()):
            if color.get(v, 0) == 1:
                return True
            if color.get(v, 0) == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    return any(color.get(u, 0) == 0 and dfs(u) for u in wait_for)


def bankers_safe(avail: list[int], max_need: list[list[int]], alloc: list[list[int]]) -> bool:
    """Classic Banker's safety: exists an order where every process can
    finish with avail + returned allocations."""
    work = list(avail)
    finish = [False] * len(max_need)
    progress = True
    while progress:
        progress = False
        for i in range(len(max_need)):
            if finish[i]:
                continue
            need = [max_need[i][r] - alloc[i][r] for r in range(len(avail))]
            if all(need[r] <= work[r] for r in range(len(avail))):
                work = [work[r] + alloc[i][r] for r in range(len(avail))]
                finish[i] = True
                progress = True
    return all(finish)


def _oracle_safe(avail: list[int], max_need: list[list[int]], alloc: list[list[int]]) -> bool:
    n = len(max_need)
    for perm in itertools.permutations(range(n)):
        work = list(avail)
        ok = True
        for i in perm:
            need = [max_need[i][r] - alloc[i][r] for r in range(len(avail))]
            if not all(need[r] <= work[r] for r in range(len(avail))):
                ok = False
                break
            work = [work[r] + alloc[i][r] for r in range(len(avail))]
        if ok:
            return True
    return False


def bench_deadlock_detect(seed: int = 20261231 + 373) -> dict[str, float]:
    rng = random.Random(seed)
    found = clean = agree = 0
    trials = 30
    for _ in range(trials):
        # plant a cycle
        n = rng.randrange(3, 7)
        wf: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            wf[i].add((i + 1) % n)
        found += int(has_deadlock(wf))
        # acyclic graph → no deadlock
        wf2: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    wf2[i].add(j)
        clean += int(not has_deadlock(wf2))
        # banker's vs exhaustive oracle (n ≤ 4)
        m = 2
        p = 4
        alloc = [[rng.randrange(0, 3) for _ in range(m)] for _ in range(p)]
        max_need = [[alloc[i][r] + rng.randrange(0, 3) for r in range(m)] for i in range(p)]
        avail = [rng.randrange(0, 4) for _ in range(m)]
        agree += int(bankers_safe(avail, max_need, alloc) == _oracle_safe(avail, max_need, alloc))
    return {
        "synthetic_cycle_detected": float(found / trials),
        "synthetic_acyclic_clean": float(clean / trials),
        "synthetic_bankers_oracle": float(agree / trials),
    }
