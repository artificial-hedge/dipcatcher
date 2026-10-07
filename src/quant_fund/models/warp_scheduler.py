"""GPU warp scheduler: greedy oldest-ready-first issue per cycle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 686


def schedule(instr: list[list[int]], warp_latency: int = 4) -> list[list[int]]:
    """instr[warp] = list of latency-costs; returns per-cycle issued warps."""
    n = len(instr)
    pos = np.zeros(n, dtype=int)
    ready = np.zeros(n, dtype=int)  # earliest cycle each warp can issue
    done = [len(x) for x in instr]
    issue: list[list[int]] = []
    cycle = 0
    while any(pos[w] < done[w] for w in range(n)):
        elig = [w for w in range(n) if pos[w] < done[w] and ready[w] <= cycle]
        if elig:
            w = min(elig)  # greedy oldest warp (lowest id = oldest)
            ready[w] = cycle + instr[w][pos[w]]
            pos[w] += 1
            issue.append([w])
        else:
            issue.append([])
        cycle += 1
    return issue


def bench_warp_scheduler(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = int(rng.randint(2, 6))
        instr = [list(rng.randint(1, 5, int(rng.randint(1, 5)))) for _ in range(n)]
        out = schedule(instr)
        issued = sum(len(x) for x in out)
        total = sum(len(x) for x in instr)
        # invariant: all instructions issued, at most one per cycle
        ok += float(issued == total and all(len(x) <= 1 for x in out))
    return {"synthetic_warp_issues_all": ok / trials}
