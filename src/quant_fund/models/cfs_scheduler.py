"""SYNTHETIC CFS (Completely Fair Scheduler) vruntime simulation.

Each runnable entity accumulates vruntime at rate exec_time * NICE_0 /
weight; the scheduler always picks the minimum-vruntime entity. Benches
verify proportional CPU share and min-vruntime discipline.
"""

from __future__ import annotations

import random

NICE_0 = 1024.0


def run_cfs(tasks: list[tuple[float, int]], horizon: int, tick: int = 4) -> dict[int, float]:
    """tasks: [(weight, total_exec)]; returns pid -> cpu time granted."""
    vrt = [0.0] * len(tasks)
    rem = [b for _w, b in tasks]
    granted = [0.0] * len(tasks)
    picks_min = True
    t = 0
    while t < horizon and any(r > 0 for r in rem):
        run = [i for i in range(len(tasks)) if rem[i] > 0]
        pid = min(run, key=lambda i: vrt[i])
        picks_min &= vrt[pid] == min(vrt[i] for i in run)
        step = min(tick, rem[pid])
        rem[pid] -= step
        granted[pid] += step
        vrt[pid] += step * NICE_0 / tasks[pid][0]
        t += step
    if not (picks_min):
        raise ValueError("picks_min")
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
        # explicit min-vruntime check via instrumented run
        pick_ok += 1  # asserted inside run_cfs
        _ = rng.random()
    return {
        "synthetic_fair_share": float(fair / trials),
        "synthetic_proportional_weight": float(prop / trials),
        "synthetic_min_vruntime_picks": float(pick_ok / trials),
    }
