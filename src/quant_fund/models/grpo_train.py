"""Group Relative Policy Optimization (DeepSeekMath / R1 style) (SYNTHETIC).

For each context sample G candidate actions from the policy; advantage
= (r_i − mean(r_group)) / std — no critic needed. Policy improves true
reward per iteration and beats the frozen reference by a wide margin.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._align_synth import (
    DX,
    N_ACT,
    action_embeddings,
    contexts,
    true_reward,
)


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("grpo_train needs the torch `nn` extra") from exc


def bench_grpo_train(
    seed: int = 251,
    n_ctx: int = 200,
    group: int = 8,
    iters: int = 200,
    beta_kl: float = 0.05,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = action_embeddings(rng)
    x = contexts(n_ctx, rng)
    r = torch.tensor(true_reward(x, emb)).float()
    x_t = torch.tensor(x).float()

    W = torch.nn.Parameter(torch.randn(DX, N_ACT) * 0.1)
    W_ref = W.detach().clone()
    opt = torch.optim.Adam([W], lr=1e-2)
    r0 = float(r.mean())
    for _i in range(iters):
        logits = x_t @ W / 0.3
        probs = torch.softmax(logits, -1)
        a_s = torch.multinomial(probs, group, replacement=True)
        r_s = torch.gather(r, 1, a_s)
        adv = (r_s - r_s.mean(1, keepdim=True)) / (r_s.std(1, keepdim=True) + 1e-6)
        logp = torch.log_softmax(logits, -1).gather(1, a_s)
        kl = torch.log_softmax(logits, -1) - torch.log_softmax(x_t @ W_ref / 0.3, -1)
        kl_pen = (torch.softmax(logits, -1) * kl).sum(-1).mean()
        loss = -(logp * adv).mean() + beta_kl * kl_pen
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = x_t @ W / 0.3
        a_best = logits.argmax(-1)
        r_final = r.gather(1, a_best[:, None]).mean().item()
        r_ref = r.gather(1, (x_t @ W_ref / 0.3).argmax(-1)[:, None]).mean().item()
        r_max = r.max(-1).values.mean().item()
    return {
        "synthetic_grpo_reward": float(r_final),
        "synthetic_grpo_ref_reward": float(r_ref),
        "synthetic_grpo_random_reward": r0,
        "synthetic_grpo_oracle_reward": float(r_max),
        "synthetic_grpo_gain": float(r_final - r0),
        "synthetic_torch_available": 1.0,
    }
