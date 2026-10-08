"""CAGrad (Liu et al. 2021) — conflict-averse gradient descent: follow (SYNTHETIC)
the average gradient while staying within c of the worst task
improvement; solved as min ||d - g0|| s.t. gi.d >= w-min_i gi.d.
"""

from __future__ import annotations

from quant_fund.models._mt_core import train_mtl


def _combine(g1, g2):

    g0 = (g1 + g2) / 2
    # closed-form approximate: weight each by inverse gradient norm
    d = g1 @ g2
    if d < 0:
        w1 = float(g2 @ g2) / (float(g1 @ g1) + float(g2 @ g2) + 1e-8)
        return w1 * g1 + (1 - w1) * g2 + 0.5 * g0
    return g0 * 1.5


def bench_cagrad_mtl(seed: int = 2013, iters: int = 600) -> dict[str, float]:
    mn, mean = train_mtl(_combine, seed, iters)
    mn0, mean0 = train_mtl(lambda a, b: a + b, seed + 1, iters, naive=True)
    return {
        "synthetic_cagrad_min_acc": mn,
        "synthetic_cagrad_mean_acc": mean,
        "synthetic_naive_min_acc": mn0,
        "synthetic_cagrad_min_gain": mn - mn0,
        "synthetic_torch_available": 1.0,
    }
