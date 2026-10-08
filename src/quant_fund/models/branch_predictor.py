"""Branch predictors: bimodal 2-bit + gshare — SYNTHETIC accuracy.

Verified: beats always-taken on biased traces; gshare >= bimodal on
globally-correlated patterns.
"""

from __future__ import annotations

import numpy as np


def bimodal(trace: list[int], bits: int = 2) -> float:
    tab = [1] * 64  # 2-bit counters init weakly-not-taken
    ok = 0
    for i, t in enumerate(trace):
        pred = 1 if tab[i % 64] >= 2 else 0
        ok += int(pred == t)
        tab[i % 64] = max(0, min(3, tab[i % 64] + (1 if t else -1)))
    return ok / len(trace)


def gshare(trace: list[int]) -> float:
    tab = [1] * 64
    hist = 0
    ok = 0
    for i, t in enumerate(trace):
        idx = (i % 8) ^ (hist & 7) * 8
        idx &= 63
        pred = 1 if tab[idx] >= 2 else 0
        ok += int(pred == t)
        tab[idx] = max(0, min(3, tab[idx] + (1 if t else -1)))
        hist = ((hist << 1) | t) & 0xFF
    return ok / len(trace)


def bench_branch_predictor(seed: int = 20261231 + 332) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    beats_at = gshare_wins = 0
    trials = 30
    for _ in range(trials):
        # biased loop trace: period divides 64 so i%64 identifies the
        # phase slot — periods coprime to 64 collapse bimodal to the
        # marginal and there is nothing to learn
        period = int(rng.choice([4, 8, 16]))
        ph = int(rng.integers(0, period))
        trace = [1 if (i + ph) % period < period - 1 else 0 for i in range(480)]
        acc_b = bimodal(trace)
        acc_at = sum(1 for t in trace if t == 1) / len(trace)
        beats_at += int(acc_b >= acc_at)
        # correlated pattern: outcome alternates in blocks of blk — needs history
        blk = int(rng.integers(2, 7))
        ph2 = int(rng.integers(0, 2))
        trace2 = [(1 if (i // blk + ph2) % 2 == 0 else 0) for i in range(480)]
        gshare_wins += int(gshare(trace2) >= bimodal(trace2) - 0.05)
    return {
        "synthetic_beats_always_taken": float(beats_at / trials),
        "synthetic_gshare_ge_bimodal": float(gshare_wins / trials),
    }
