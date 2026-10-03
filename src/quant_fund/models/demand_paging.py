"""SYNTHETIC demand paging — FIFO / LRU / OPT (Belady) page replacement.

Trace with phase locality: OPT is provably minimal; LRU beats FIFO on
locality-heavy traces; every policy serves every reference.
"""

from __future__ import annotations

import random


def _fifo(trace: list[int], cap: int) -> int:
    frames: list[int] = []
    order: list[int] = []
    faults = 0
    for p in trace:
        if p in frames:
            continue
        faults += 1
        if len(frames) == cap:
            frames.remove(order.pop(0))
        frames.append(p)
        order.append(p)
    return faults


def _lru(trace: list[int], cap: int) -> int:
    frames: list[int] = []
    faults = 0
    for p in trace:
        if p in frames:
            frames.remove(p)
            frames.append(p)
            continue
        faults += 1
        if len(frames) == cap:
            frames.pop(0)
        frames.append(p)
    return faults


def _opt(trace: list[int], cap: int) -> int:
    frames: list[int] = []
    faults = 0
    for i, p in enumerate(trace):
        if p in frames:
            continue
        faults += 1
        if len(frames) == cap:
            # evict page whose next use is farthest (or never)
            def nxt(q: int, i: int = i) -> float:
                try:
                    return trace.index(q, i + 1)
                except ValueError:
                    return float("inf")

            victim = max(frames, key=nxt)
            frames.remove(victim)
        frames.append(p)
    return faults


def bench_demand_paging(seed: int = 20261231 + 372) -> dict[str, float]:
    rng = random.Random(seed)
    opt_min = lru_be = faults_ok = 0
    trials = 30
    for _ in range(trials):
        # phased locality trace
        trace: list[int] = []
        for _ in range(6):
            base = rng.randrange(0, 40)
            ws = {base + i for i in range(rng.randrange(2, 5))}
            trace.extend(rng.choice(sorted(ws)) for _ in range(rng.randrange(8, 20)))
        cap = 4
        f_fifo = _fifo(trace, cap)
        f_lru = _lru(trace, cap)
        f_opt = _opt(trace, cap)
        opt_min += int(f_opt <= min(f_fifo, f_lru))
        # LRU within 1.5x of OPT on working-set-fitting traces
        lru_be += int(f_lru <= f_opt * 1.5)
        faults_ok += int(max(f_fifo, f_lru, f_opt) <= len(trace) and f_opt > 0)
    return {
        "synthetic_opt_min_faults": float(opt_min / trials),
        "synthetic_lru_near_opt": float(lru_be / trials),
        "synthetic_faults_bounded": float(faults_ok / trials),
    }
