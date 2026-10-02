"""Nash-MTL (Navon et al. 2022) — combined direction proportional to
task-gradient terms weighted by inverse of their contribution to the
log barrier: d solves sum_i (1/d^T g_i-scaled) — closed-form 2-task
approx via equal-angle bisector weighting.
"""

from __future__ import annotations

from quant_fund.models._mt_core import train_mtl


def _combine(g1, g2):

    # Nash bargaining direction ≈ normalize each, then min-norm mix
    u1 = g1 / (g1.norm() + 1e-8)
    u2 = g2 / (g2.norm() + 1e-8)
    a = float((u2 @ u2 - u1 @ u2) / ((u1 - u2) @ (u1 - u2) + 1e-8))
    a = max(0.0, min(1.0, a))
    return a * u1 + (1 - a) * u2


def bench_nash_mtl(seed: int = 2027, iters: int = 600) -> dict[str, float]:
    mn, mean = train_mtl(_combine, seed, iters)
    mn0, mean0 = train_mtl(lambda a, b: a + b, seed + 1, iters, naive=True)
    return {
        "synthetic_nash_min_acc": mn,
        "synthetic_nash_mean_acc": mean,
        "synthetic_naive_min_acc": mn0,
        "synthetic_nash_min_gain": mn - mn0,
        "torch_available": 1.0,
    }
