"""ENAS-style RL controller (Pham et al. 2018).

REINFORCE over arch genes (width/depth/act), reward = val acc;
moving-average baseline. Final sampled arch acc vs uniform prior.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import split
from quant_fund.models._nas_synth import _torch, eval_arch, noisy_labels


def bench_enas_controller(
    seed: int = 479,
    n: int = 300,
    steps: int = 40,
    iters: int = 40,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(noisy_labels(y_tr, seed))
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    # logits over genes: width(5), depth(3), act(2)
    logit_w = np.zeros(5)
    logit_d = np.zeros(3)
    logit_a = np.zeros(2)
    lr = 0.25
    baseline = 0.0
    best = 0.0
    rewards = []
    for _i in range(steps):
        pw = np.exp(logit_w - logit_w.max())
        pw /= pw.sum()
        pd = np.exp(logit_d - logit_d.max())
        pd /= pd.sum()
        pa = np.exp(logit_a - logit_a.max())
        pa /= pa.sum()
        w = int(rng.choice(5, p=pw))
        d = int(rng.choice(3, p=pd))
        a = int(rng.choice(2, p=pa))
        arch = (int([4, 8, 16, 24, 48][w]), int(d + 1), a)
        r = eval_arch(arch, x_tr_t, y_tr_t, x_te_t, y_te_t, iters, seed=hash(arch) % 1000)
        adv = r - baseline
        baseline = 0.9 * baseline + 0.1 * r
        # REINFORCE on each gene
        onehot = np.eye(5)[w]
        logit_w += lr * adv * (onehot - pw)
        onehot = np.eye(3)[d]
        logit_d += lr * adv * (onehot - pd)
        onehot = np.eye(2)[a]
        logit_a += lr * adv * (onehot - pa)
        best = max(best, r)
        rewards.append(r)
    # converged policy's argmax arch
    arch_final = (
        int([4, 8, 16, 24, 48][int(np.argmax(logit_w))]),
        int(np.argmax(logit_d) + 1),
        int(np.argmax(logit_a)),
    )
    acc_final = eval_arch(arch_final, x_tr_t, y_tr_t, x_te_t, y_te_t, iters, seed=11)
    acc_prior = float(np.mean(rewards[:6]))  # near-uniform early samples
    return {
        "synthetic_ec_final_acc": acc_final,
        "synthetic_ec_prior_acc": acc_prior,
        "synthetic_ec_best": best,
        "synthetic_ec_gain": acc_final - acc_prior,
        "torch_available": 1.0,
    }
