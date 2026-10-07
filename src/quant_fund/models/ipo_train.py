"""Identity Preference Optimization (IPO / ΨPO, Azar et al. 2024).

DPO's logistic objective saturates under deterministic preferences and
overfits label noise; IPO regresses the log-ratio gap to a fixed target
((log πθ/πref)(w) − (log πθ/πref)(l) − 1/(2β))². Compared to DPO on
the same noisy prefs at higher label-noise — the documented IPO edge.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._align_synth import (
    DX,
    N_ACT,
    action_embeddings,
    best_action_rate,
    contexts,
    pref_pairs,
)


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("ipo_train needs the torch `nn` extra") from exc


def bench_ipo_train(
    seed: int = 239,
    n_pairs: int = 400,
    n_eval: int = 300,
    beta: float = 0.2,
    noise: float = 0.4,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x, a_w, a_l = pref_pairs(n_pairs, emb, rng, noise=noise)
    x_e = contexts(n_eval, rng)

    def phi(x, a):
        n = x.shape[0]
        f = np.zeros((n, N_ACT * DX))
        f[np.arange(n)[:, None], a[:, None] * DX + np.arange(DX)[None, :]] = x
        return torch.tensor(f).float()

    fw = phi(x, a_w)
    fl = phi(x, a_l)
    ref = torch.randn(N_ACT * DX) * 0.1

    def train(mode):
        theta = torch.nn.Parameter(torch.zeros(N_ACT * DX))
        opt = torch.optim.Adam([theta], lr=5e-3)
        for _i in range(iters):
            gap = ((fw @ theta - fw @ ref) - (fl @ theta - fl @ ref)) * beta
            if mode == "dpo":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    gap, torch.ones_like(gap)
                )
            else:  # ipo
                loss = ((gap - 1.0 / (2 * beta)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        th = theta.detach().numpy().reshape(N_ACT, DX)
        return best_action_rate(x_e @ th.T, x_e, emb)

    acc_dpo = train("dpo")
    acc_ipo = train("ipo")
    return {
        "synthetic_ipo_best_rate": acc_ipo,
        "synthetic_ipo_dpo_rate": acc_dpo,
        "synthetic_ipo_gain_vs_dpo": acc_ipo - acc_dpo,
        "synthetic_torch_available": 1.0,
    }
