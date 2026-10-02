"""PCGrad (Yu et al. 2020) — project conflicting gradients: when
g1.g2 < 0, subtract the mutual projection from both. Min-task acc
vs naive-sum SGD on the conflicting two-task fixture.
"""

from __future__ import annotations

from quant_fund.models._mt_core import train_mtl


def _combine(g1, g2):

    d = float(g1 @ g2)
    if d < 0:
        g1c = g1 - d * g2 / (g2 @ g2 + 1e-8)
        g2c = g2 - d * g1 / (g1 @ g1 + 1e-8)
    else:
        g1c, g2c = g1, g2
    return g1c + g2c


def bench_pcgrad(seed: int = 2001, iters: int = 600) -> dict[str, float]:
    mn, mean = train_mtl(_combine, seed, iters)
    mn0, mean0 = train_mtl(lambda a, b: a + b, seed + 1, iters, naive=True)
    return {
        "synthetic_pcgrad_min_acc": mn,
        "synthetic_pcgrad_mean_acc": mean,
        "synthetic_naive_min_acc": mn0,
        "synthetic_naive_mean_acc": mean0,
        "synthetic_pcgrad_min_gain": mn - mn0,
        "torch_available": 1.0,
    }
