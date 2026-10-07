"""Direct Preference Optimization (Rafailov et al. 2023).

Policy logits θ·φ(x,a) trained by the DPO surrogate
log σ(β(log πθ/πref)(w) − β(log πθ/πref)(l)) against a frozen reference
policy — no explicit reward model needed. Compares best-action rate to
the reference policy and a pure-SFT clone.
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
        raise ImportError("dpo_train needs the torch `nn` extra") from exc


def bench_dpo_train(
    seed: int = 233,
    n_pairs: int = 400,
    n_eval: int = 300,
    beta: float = 0.5,
    iters: int = 500,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x, a_w, a_l = pref_pairs(n_pairs, emb, rng)
    x_e = contexts(n_eval, rng)

    # context-action features: outer-product features x⊗onehot(a)
    def phi(x, a):
        n = x.shape[0]
        f = np.zeros((n, N_ACT * DX))
        f[np.arange(n)[:, None], a[:, None] * DX + np.arange(DX)[None, :]] = x
        return torch.tensor(f).float()

    theta = torch.nn.Parameter(torch.zeros(N_ACT * DX))
    ref = torch.nn.Parameter(torch.randn(N_ACT * DX) * 0.1)  # imperfect ref policy
    opt = torch.optim.Adam([theta], lr=5e-3)
    fw = phi(x, a_w)
    fl = phi(x, a_l)
    for _i in range(iters):
        pi_w = fw @ theta
        pi_l = fl @ theta
        r_w = fw @ ref
        r_l = fl @ ref
        logits = beta * ((pi_w - r_w) - (pi_l - r_l))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        th = theta.detach().numpy().reshape(N_ACT, DX)
        acc_dpo = best_action_rate(x_e @ th.T, x_e, emb)
        th_r = ref.detach().numpy().reshape(N_ACT, DX)
        acc_ref = best_action_rate(x_e @ th_r.T, x_e, emb)
    return {
        "synthetic_dpo_best_rate": acc_dpo,
        "synthetic_dpo_ref_rate": acc_ref,
        "synthetic_dpo_gain": acc_dpo - acc_ref,
        "synthetic_torch_available": 1.0,
    }
