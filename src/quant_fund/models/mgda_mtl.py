"""MGDA (Desideri 2012) — multiple-gradient descent: min-norm point in
the convex hull of task gradients via Franke-Wolfe; the direction
decreases all tasks simultaneously.
"""

from __future__ import annotations

from quant_fund.models._mt_core import train_mtl


def _combine(g1, g2):
    # min-norm convex combination: min ||a g1 + (1-a) g2||
    a = float((g2 @ g2 - g1 @ g2) / ((g1 - g2) @ (g1 - g2) + 1e-8))
    a = max(0.0, min(1.0, a))
    return a * g1 + (1 - a) * g2


def bench_mgda_mtl(seed: int = 2007, iters: int = 600) -> dict[str, float]:
    mn, mean = train_mtl(_combine, seed, iters)
    mn0, mean0 = train_mtl(lambda a, b: a + b, seed + 1, iters, naive=True)
    return {
        "synthetic_mgda_min_acc": mn,
        "synthetic_mgda_mean_acc": mean,
        "synthetic_naive_min_acc": mn0,
        "synthetic_mgda_min_gain": mn - mn0,
        "torch_available": 1.0,
    }
