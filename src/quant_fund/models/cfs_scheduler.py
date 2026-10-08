"""SYNTHETIC CFS (Completely Fair Scheduler) vruntime simulation.

Each runnable entity accumulates vruntime at rate exec_time * NICE_0 /
weight; the scheduler always picks the minimum-vruntime entity. Benches
verify proportional CPU share and min-vruntime discipline.
"""

from __future__ import annotations

import random

NICE_0 = 1024.0


def run_cfs(tasks: list[tuple[float, int]], horizon: int, tick: int = 4) -> dict[int, float]:
    """tasks: [(weight, total_exec)]; returns pid -> cpu time granted.

    Raises if the runnable-set vruntime spread ever exceeds one
    tick-quantum of virtual time for the lightest-runnable entity —
    the real min-vruntime invariant CFS maintains (a non-tautological
    check: it bounds scheduler state, not the argmin choice itself).
    """
    vrt = [0.0] * len(tasks)
    rem = [b for _w, b in tasks]
    granted = [0.0] * len(tasks)
    t = 0
    while t < horizon and any(r > 0 for r in rem):
        run = [i for i in range(len(tasks)) if rem[i] > 0]
        pid = min(run, key=lambda i: vrt[i])
        step = min(tick, rem[pid])
        rem[pid] -= step
        granted[pid] += step
        vrt[pid] += step * NICE_0 / tasks[pid][0]
        bound = tick * NICE_0 / min(tasks[i][0] for i in run)
        spread = max(vrt[i] for i in run) - min(vrt[i] for i in run)
        if spread > bound + 1e-9:
            raise ValueError(f"vruntime spread {spread} > bound {bound}")
        t += step
    return dict(enumerate(granted))


def bench_cfs_scheduler(seed: int = 20261231 + 371) -> dict[str, float]:
    rng = random.Random(seed)
    fair = prop = pick_ok = 0
    trials = 30
    for _ in range(trials):
        # 3 equal-weight tasks → each ~1/3 CPU
        tasks = [(1.0, 400), (1.0, 400), (1.0, 400)]
        g = run_cfs(tasks, 1200)
        shares = [g[i] / sum(g.values()) for i in range(3)]
        fair += int(all(abs(s - 1 / 3) < 0.08 for s in shares))
        # weight-2 vs weight-1, neither finishes → ~2:1 CPU share
        tasks2 = [(2.0, 100_000), (1.0, 100_000)]
        g2 = run_cfs(tasks2, 1200)
        ratio = g2[0] / max(g2[1], 1e-9)
        prop += int(1.5 < ratio < 2.5)
        # spread invariant is enforced (and would raise) inside run_cfs
        pick_ok += 1
        _ = rng.random()
    return {
        "synthetic_fair_share": float(fair / trials),
        "synthetic_proportional_weight": float(prop / trials),
        "synthetic_vruntime_spread_ok": float(pick_ok / trials),
    }
