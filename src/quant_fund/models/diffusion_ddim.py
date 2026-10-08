"""DDIM deterministic sampling (Song et al. 2021) (SYNTHETIC).

Toy 2-point "image" distribution; compare DDIM (η=0, k steps) vs
ancestral DDPM at equal step budget — FID-proxy (W1) gap.
"""

from __future__ import annotations

import numpy as np


def _fwd(x0, t_idx, alphas_cum, rng):
    a = alphas_cum[t_idx]
    eps = rng.normal(0, 1, x0.shape)
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps, eps


def bench_diffusion_ddim(
    seed: int = 541,
    n: int = 400,
    steps: int = 5,
    t_total: int = 100,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # data: mixture at ±1.5 (the "image" is a scalar)
    x0 = rng.choice([-1.5, 1.5], n) + rng.normal(0, 0.15, n)
    betas = np.linspace(1e-4, 0.02, t_total)
    alphas = 1 - betas
    ac = np.cumprod(alphas)

    # "trained model": eps_hat ≈ true eps (use closed-form denoise mean)
    def denoise(xt, t, ddim):
        a_t = ac[t]
        a_prev = ac[t - 1] if t > 0 else 1.0
        # model eps = score approx: predict eps from xt via shrinkage
        # closed-form posterior mean for the 2-point mixture
        z = xt / np.sqrt(a_t)
        x0_hat = 1.5 * np.tanh(1.5 * z / (1 - a_t + 0.15))
        eps = (xt - np.sqrt(a_t) * x0_hat) / np.sqrt(1 - a_t)
        if ddim:
            sigma = 0.0
        else:
            sigma = np.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        coef = np.sqrt(1 - a_prev - sigma**2) if ddim else np.sqrt(1 - a_prev)
        noise = 0 if ddim else rng.normal(0, 1, xt.shape)
        return np.sqrt(a_prev) * x0_hat + coef * eps + sigma * noise

    ts = np.linspace(t_total - 1, 0, steps).astype(int)
    for ddim in (False, True):
        xt = rng.normal(0, 1, n)
        for t in ts:
            xt = denoise(xt, int(t), ddim)
        # W1 to data
        w = np.abs(np.sort(x0) - np.sort(xt)).mean()
        if ddim:
            w_ddim = float(w)
        else:
            w_ddpm = float(w)
    return {
        "synthetic_ddim_w1": w_ddim,
        "synthetic_ddpm_w1": w_ddpm,
        "synthetic_ddim_gain": w_ddpm - w_ddim,
    }
