"""Kahneman-Tversky Optimization (Ethayarajh et al. 2024) (SYNTHETIC).

Binary desirable/undesirable labels instead of pairs. KTO loss uses
value-function asymmetry: λ_D σ(β(z_ref − r_θ)) on good examples,
λ_U σ(β(r_θ − z_ref)) on bad, with z_ref the KL anchor. Compared with
SFT-only baseline on best-action rate.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._align_synth import (
    DX,
    N_ACT,
    action_embeddings,
    best_action_rate,
    contexts,
    true_reward,
)


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("kto_train needs the torch `nn` extra") from exc


def bench_kto_train(
    seed: int = 241,
    n_lab: int = 500,
    n_eval: int = 300,
    beta: float = 0.5,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x = contexts(n_lab, rng)
    a = rng.integers(0, N_ACT, n_lab)
    r = true_reward(x, emb)
    good = r[np.arange(n_lab), a] > np.median(r, axis=-1)
    x_e = contexts(n_eval, rng)

    def phi(x, a):
        n = x.shape[0]
        f = np.zeros((n, N_ACT * DX))
        f[np.arange(n)[:, None], a[:, None] * DX + np.arange(DX)[None, :]] = x
        return torch.tensor(f).float()

    fa = phi(x, a)
    ref = torch.zeros(N_ACT * DX)
    theta = torch.nn.Parameter(torch.zeros(N_ACT * DX))
    opt = torch.optim.Adam([theta], lr=5e-3)
    g_t = torch.tensor(good.astype(float))
    for _i in range(iters):
        r_θ = beta * (fa @ theta - fa @ ref)
        pos = torch.nn.functional.binary_cross_entropy_with_logits(
            r_θ, torch.ones_like(r_θ), reduction="none"
        )
        neg = torch.nn.functional.binary_cross_entropy_with_logits(
            -r_θ, torch.ones_like(r_θ), reduction="none"
        )
        loss = (g_t * pos + (1 - g_t) * neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        th = theta.detach().numpy().reshape(N_ACT, DX)
        acc = best_action_rate(x_e @ th.T, x_e, emb)
        acc_rand = 1.0 / N_ACT
    return {
        "synthetic_kto_best_rate": acc,
        "synthetic_kto_random_rate": acc_rand,
        "synthetic_kto_gain": acc - acc_rand,
        "synthetic_torch_available": 1.0,
    }
