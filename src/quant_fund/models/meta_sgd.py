"""Meta-SGD (Li et al. 2017) — learn a per-parameter adaptive learning
rate/direction vector α alongside the init: θ' = θ − α ⊙ ∇L. Query MSE
vs fixed-LR MAML-style init.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("meta_sgd requires torch (pip install -e .[nn])") from exc
    return torch


def bench_meta_sgd(seed: int = 867, n_tasks: int = 30, K: int = 5) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    w1 = torch.nn.Parameter(torch.randn(1, 32) * 0.1)
    b1 = torch.nn.Parameter(torch.zeros(32))
    w2 = torch.nn.Parameter(torch.randn(32, 1) * 0.1)
    b2 = torch.nn.Parameter(torch.zeros(1))
    theta = [w1, b1, w2, b2]
    alpha = [torch.nn.Parameter(torch.full(t.shape, 0.05)) for t in theta]
    opt = torch.optim.Adam(theta + alpha, lr=0.005)

    def f(xv, ps):
        h = torch.tanh(xv @ ps[0] + ps[1])
        return (h @ ps[2] + ps[3]).squeeze(-1)

    for _t in range(n_tasks):
        xs, ys, xq, yq = sine_task(rng, K=K)
        Xs = torch.tensor(xs).float()[:, None]
        pred = f(Xs, theta)
        loss_i = ((pred - torch.tensor(ys).float()) ** 2).mean()
        grads = torch.autograd.grad(loss_i, theta, create_graph=False)
        theta_i = [t - a * g for t, a, g in zip(theta, alpha, grads, strict=True)]
        pred_q = f(torch.tensor(xq).float()[:, None], theta_i)
        loss = ((pred_q - torch.tensor(yq).float()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    mses = []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 5000 + i), K=K)
        Xs = torch.tensor(xs).float()[:, None]
        ti = [t.detach().clone().requires_grad_(True) for t in theta]
        for _ in range(5):
            pred = f(Xs, ti)
            loss_i = ((pred - torch.tensor(ys).float()) ** 2).mean()
            grads = torch.autograd.grad(loss_i, ti)
            ti = [t - a * g for t, a, g in zip(ti, alpha, grads, strict=True)]
        with torch.no_grad():
            pred = f(torch.tensor(xq).float()[:, None], ti).numpy()
        mses.append(float(((pred - yq) ** 2).mean()))
    # fixed-LR baseline: same init, scalar lr 0.01
    mses_b = []
    for i in range(8):
        xs, ys, xq, yq = sine_task(np.random.default_rng(seed + 5000 + i), K=K)
        Xs = torch.tensor(xs).float()[:, None]
        ti = [t.detach().clone().requires_grad_(True) for t in theta]
        for _ in range(5):
            pred = f(Xs, ti)
            loss_i = ((pred - torch.tensor(ys).float()) ** 2).mean()
            grads = torch.autograd.grad(loss_i, ti)
            ti = [t - 0.01 * g for t, g in zip(ti, grads, strict=True)]
        with torch.no_grad():
            pred = f(torch.tensor(xq).float()[:, None], ti).numpy()
        mses_b.append(float(((pred - yq) ** 2).mean()))
    return {
        "synthetic_msgd_query_mse": float(np.mean(mses)),
        "synthetic_msgd_fixedlr_mse": float(np.mean(mses_b)),
        "synthetic_msgd_gain": float(np.mean(mses_b) - np.mean(mses)),
        "synthetic_torch_available": 1.0,
    }
