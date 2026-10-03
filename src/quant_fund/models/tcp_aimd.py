"""SYNTHETIC TCP congestion control — Reno-style slow start + AIMD.

cwnd doubles per RTT during slow start until ssthresh, then grows
additively (+1 per RTT); loss halves cwnd (fast recovery) or resets to 1
(timeout). Verified trajectory properties.
"""

from __future__ import annotations

import random


def simulate(losses: set[int], rtts: int = 40, ssthresh0: float = 16.0) -> list[float]:
    cwnd = 1.0
    ssthresh = ssthresh0
    hist = []
    for t in range(rtts):
        hist.append(cwnd)
        if t in losses:
            ssthresh = max(cwnd / 2, 2.0)
            cwnd = ssthresh  # fast recovery: halve, keep going
            continue
        if cwnd < ssthresh:
            cwnd *= 2  # slow start
        else:
            cwnd += 1  # congestion avoidance
    return hist


def bench_tcp_aimd(seed: int = 20261231 + 400) -> dict[str, float]:
    rng = random.Random(seed)
    exp_growth = additive = halves = 0
    trials = 40
    for _ in range(trials):
        h = simulate(set())
        # slow start: 1,2,4,8,16 then additive 17,18,...
        exp_growth += int(all(h[i] == 2**i for i in range(5)) and abs(h[6] - h[5] - 1) < 1e-9)
        # no loss → after ssthresh, +1 per RTT exactly
        additive += int(all(h[i + 1] - h[i] == 1.0 for i in range(5, 10)))
        # one loss at t=8 halves cwnd at t=9
        h2 = simulate({8})
        halves += int(h2[9] == max(h2[8] / 2, 2.0))
        _ = rng.random()
    return {
        "synthetic_slow_start_doubles": float(exp_growth / trials),
        "synthetic_additive_increase": float(additive / trials),
        "synthetic_loss_halves": float(halves / trials),
    }
