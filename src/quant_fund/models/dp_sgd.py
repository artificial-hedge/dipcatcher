"""DP-SGD — per-sample gradient clipping + Gaussian noise (Abadi 2016) (SYNTHETIC).

Trains the shared MLP with per-example gradients clipped to C and
Gaussian noise σ·C added to the batch gradient — the privacy-utility
curve: accuracy vs noise multiplier σ on the synth task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import acc_of, make_mlp, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("dp_sgd requires torch (pip install -e .[nn])") from exc
    return torch


def _train_dp(torch, x, y, sigma, clip=1.0, iters=60, lr=0.02, seed=0):
    torch.manual_seed(seed)
    net = make_mlp(torch)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    n = len(x)
    for _i in range(iters):
        # per-sample grads
        total = [torch.zeros_like(p) for p in net.parameters()]
        for j in range(n):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x[j : j + 1]), y[j : j + 1])
            loss.backward()
            for k, p in enumerate(net.parameters()):
                g = p.grad.detach().clone()
                total[k] += g / max(1.0, float(g.norm() / clip))
        for k, p in enumerate(net.parameters()):
            noisy = total[k] / n + sigma * clip * torch.randn_like(total[k]) / n
            p.data -= lr * noisy
    return net


def bench_dp_sgd(
    seed: int = 419,
    n: int = 400,
    iters: int = 40,
    n_sub: int = 80,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t = torch.tensor(x_tr).float()
    y_tr_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te).float()
    y_te_t = torch.tensor(y_te)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_tr), n_sub, replace=False)
    xs = x_tr_t[idx]
    ys = y_tr_t[idx]
    base = _train_dp(torch, xs, ys, sigma=0.0, iters=iters, seed=seed)
    dp = _train_dp(torch, xs, ys, sigma=0.4, iters=iters, seed=seed + 1)
    dp2 = _train_dp(torch, xs, ys, sigma=1.2, iters=iters, seed=seed + 2)
    acc_b = acc_of(torch, base, x_te_t, y_te_t)
    acc_dp = acc_of(torch, dp, x_te_t, y_te_t)
    acc_dp2 = acc_of(torch, dp2, x_te_t, y_te_t)
    return {
        "synthetic_dpsgd_acc": acc_dp,
        "synthetic_dpsgd_nodp_acc": acc_b,
        "synthetic_dpsgd_highnoise_acc": acc_dp2,
        "synthetic_dpsgd_drop": acc_b - acc_dp,
        "synthetic_torch_available": 1.0,
    }
