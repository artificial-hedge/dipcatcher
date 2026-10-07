"""Perturb-and-MAP differentiation (Berthet et al. 2020) — the Jacobian (SYNTHETIC)
of argmax under Gumbel perturbations is estimated by Monte-Carlo:
d/ds E[argmax(s+εZ)] via the perturbed maximizer's covariance.
Selection-task gradient vs finite-diff.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._da_synth import finite_diff_grad, grad_corr, loss_np, sel_data


def bench_perturb_map(
    seed: int = 2399, trials: int = 20, M: int = 200, sig: float = 0.5
) -> dict[str, float]:
    corrs = []
    for i in range(trials):
        s0, _ = sel_data(seed + i)
        rng = np.random.default_rng(seed + 1000 + i)
        zs = sig * rng.standard_normal((M, len(s0)))
        vals = np.array([loss_np(s0 + zs[m]) for m in range(M)])
        # ∇ E[v(s+σZ)] = Cov(Z, v)/σ² (Gaussian score function)
        g_pm = ((zs / sig) * (vals - vals.mean())[:, None]).mean(0) / sig
        g_fd = finite_diff_grad(s0)
        corrs.append(grad_corr(g_pm, g_fd))
    return {
        "synthetic_pmap_grad_corr": float(np.mean(corrs)),
        "synthetic_pmap_mc_std": float(np.std(corrs)),
        "synthetic_torch_available": 0.0,
    }
