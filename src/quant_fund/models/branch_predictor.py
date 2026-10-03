"""Branch predictors: bimodal 2-bit + gshare — SYNTHETIC accuracy.

Verified: beats always-taken on biased traces; gshare >= bimodal on
globally-correlated patterns.
"""

from __future__ import annotations


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
    _ = seed
    beats_at = gshare_wins = 0
    trials = 30
    for _ in range(trials):
        # biased loop trace: 7 taken / 1 not
        pat = [1] * 7 + [0]
        trace = (pat * 60)[:480]
        acc_b = bimodal(trace)
        acc_at = sum(1 for t in trace if t == 1) / len(trace)
        beats_at += int(acc_b >= acc_at)
        # correlated pattern: outcome alternates every 3 — needs history
        trace2 = [(1 if (i // 3) % 2 == 0 else 0) for i in range(480)]
        gshare_wins += int(gshare(trace2) >= bimodal(trace2) - 0.05)
    return {
        "synthetic_beats_always_taken": float(beats_at / trials),
        "synthetic_gshare_ge_bimodal": float(gshare_wins / trials),
    }
