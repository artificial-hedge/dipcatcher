"""PPO-style RLHF with KL-to-reference penalty.

Policy maximizes a learned reward model's score. Without KL control
the policy collapses onto the RM's argmax errors (reward hacking —
high RM score, low true reward). KL-penalized PPO gets lower RM score
but HIGHER true reward — the canonical anti-overoptimization result.
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
        raise ImportError("rlhf_ppo needs the torch `nn` extra") from exc


def bench_rlhf_ppo(
    seed: int = 257,
    n_pairs: int = 120,
    n_ctx: int = 200,
    iters_rm: int = 300,
    iters_ppo: int = 600,
    beta_kl: float = 0.3,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x, a_w, a_l = pref_pairs(n_pairs, emb, rng, noise=0.2)
    x_c = contexts(n_ctx, rng)
    x_t = torch.tensor(x_c).float()
    r_true = torch.tensor(true_reward(x_c, emb)).float()

    # train reward model
    E_hat = torch.nn.Parameter(torch.randn(N_ACT, DX) * 0.1)
    opt = torch.optim.Adam([E_hat], lr=1e-2)
    x_p = torch.tensor(x).float()
    w_t, l_t = torch.tensor(a_w), torch.tensor(a_l)
    for _i in range(iters_rm):
        s = (x_p[:, None, :] * E_hat[None]).sum(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s.gather(1, w_t[:, None]) - s.gather(1, l_t[:, None]),
            torch.ones(len(x_p), 1),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    # partial-information RM: can only see first 2 of 4 context dims
    E_hat = E_hat.detach()
    E_hat[:, 2:] = 0.0

    def train(beta):
        W = torch.nn.Parameter(torch.zeros(DX, N_ACT))
        W_ref = torch.zeros(DX, N_ACT)
        opt = torch.optim.Adam([W], lr=1e-2)
        r_rm = (x_t @ E_hat.T).float()
        for _i in range(iters_ppo):
            logits = x_t @ W
            probs = torch.softmax(logits, -1)
            a_s = torch.multinomial(probs, 1).squeeze(-1)
            r_s = r_rm.gather(1, a_s[:, None]).squeeze(-1)
            adv = r_s - r_rm.mean(-1)
            logp = torch.log_softmax(logits, -1).gather(1, a_s[:, None]).squeeze(-1)
            kl = (
                (
                    torch.softmax(logits, -1)
                    * (torch.log_softmax(logits, -1) - torch.log_softmax(x_t @ W_ref, -1))
                )
                .sum(-1)
                .mean()
            )
            loss = -(logp * adv).mean() + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            a_sel = (x_t @ W).argmax(-1)
            return (
                r_rm.gather(1, a_sel[:, None]).mean().item(),
                r_true.gather(1, a_sel[:, None]).mean().item(),
            )

    rm_free, true_free = train(0.0)
    rm_kl, true_kl = train(beta_kl)
    return {
        "synthetic_rlhf_kl_true_reward": float(true_kl),
        "synthetic_rlhf_free_true_reward": float(true_free),
        "synthetic_rlhf_kl_rm_score": float(rm_kl),
        "synthetic_rlhf_free_rm_score": float(rm_free),
        "synthetic_rlhf_overopt_gap": float(true_kl - true_free),
        "synthetic_torch_available": 1.0,
    }
