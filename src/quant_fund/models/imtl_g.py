"""IMTL-G (Liu et al. 2021) — impartial multi-task learning in gradient
space: find d with equal inner products with all task gradients
(2-task closed form = projection onto the null-balance direction).
"""

from __future__ import annotations

from quant_fund.models._mt_core import train_mtl


def _combine(g1, g2):

    # equal-descent direction: d = g1 + a g2 with (g1 - g2).d = 0 → a = (g2² - g1.g2)/(g1² - g1.g2)
    denom = float(g1 @ g1 - g1 @ g2)
    numer = float(g2 @ g2 - g1 @ g2)
    a = numer / (denom + 1e-8) if abs(denom) > 1e-8 else 1.0
    a = max(0.0, min(1.0, a))
    return g1 * (1 - a) + g2 * a


def bench_imtl_g(seed: int = 2033, iters: int = 600) -> dict[str, float]:
    mn, mean = train_mtl(_combine, seed, iters)
    mn0, mean0 = train_mtl(lambda a, b: a + b, seed + 1, iters, naive=True)
    return {
        "synthetic_imtl_min_acc": mn,
        "synthetic_imtl_mean_acc": mean,
        "synthetic_naive_min_acc": mn0,
        "synthetic_imtl_min_gain": mn - mn0,
        "synthetic_torch_available": 1.0,
    }
