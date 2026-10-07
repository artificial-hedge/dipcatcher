"""Bradley-Terry reward model (Christiano et al. / InstructGPT).

R(x,a) = x·Ê_a fit by pairwise logistic loss on preference pairs.
Reported: pairwise AUC vs the true reward order and top-1 hit rate —
the learned reward's argmax matches the oracle's.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._align_synth import (
    DX,
    N_ACT,
    action_embeddings,
    contexts,
    pref_pairs,
    true_reward,
)


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("reward_model needs the torch `nn` extra") from exc


def bench_reward_model(
    seed: int = 229,
    n_pairs: int = 500,
    n_eval: int = 300,
    iters: int = 400,
    noise: float = 0.1,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x, a_w, a_l = pref_pairs(n_pairs, emb, rng, noise)
    x_e = contexts(n_eval, rng)
    r_e = true_reward(x_e, emb)

    E_hat = torch.nn.Parameter(torch.randn(N_ACT, DX) * 0.1)
    opt = torch.optim.Adam([E_hat], lr=1e-2)
    x_t = torch.tensor(x).float()
    w_t = torch.tensor(a_w)
    l_t = torch.tensor(a_l)
    for _i in range(iters):
        s_w = (x_t * E_hat[w_t]).sum(-1)
        s_l = (x_t * E_hat[l_t]).sum(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s_w - s_l, torch.ones_like(s_w))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r_hat = (x_t[:, None, :] * E_hat[None, :, :]).sum(-1).numpy()
        auc = float((r_hat[np.arange(n_pairs), a_w] > r_hat[np.arange(n_pairs), a_l]).mean())
        re_hat = (torch.tensor(x_e).float()[:, None, :] * E_hat[None, :, :]).sum(-1).numpy()
        top1 = float((re_hat.argmax(-1) == r_e.argmax(-1)).mean())
    return {
        "synthetic_rm_pair_auc": auc,
        "synthetic_rm_top1_acc": top1,
        "synthetic_rm_random_auc": 0.5,
        "synthetic_rm_top1_gain": top1 - 1.0 / N_ACT,
        "synthetic_torch_available": 1.0,
    }
