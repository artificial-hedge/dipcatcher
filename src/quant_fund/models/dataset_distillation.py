"""Dataset distillation — gradient matching (Zhao et al. 2021) (SYNTHETIC).

Learn k synthetic samples whose gradient trajectory matches real-data
training. A logistic model trained on the distillate alone recovers
most of full-data accuracy vs a random-subset coreset.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._data_synth import synth_dataset


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("dataset_distillation needs the torch `nn` extra") from exc


def bench_dataset_distillation(
    seed: int = 293,
    n: int = 600,
    k: int = 20,
    iters: int = 200,
    inner_steps: int = 5,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y, _h = synth_dataset(n, rng)
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    # learn k synthetic points by matching their CE-grad to the real grad
    x_s = torch.nn.Parameter(torch.randn(k, 8))
    y_s = torch.tensor(np.tile([0, 1], k // 2))
    opt = torch.optim.Adam([x_s], lr=5e-2)
    for _i in range(iters):
        # inner model adapts to the synthetic set for inner_steps
        # differentiable SGD steps before gradients are matched — the
        # trajectory keeps x_s in the graph (fresh random init each
        # outer iter, per Zhao et al.)
        w_lin = torch.nn.Linear(8, 2, bias=True)
        w, b = w_lin.weight, w_lin.bias
        for _ in range(inner_steps):
            l_in = torch.nn.functional.cross_entropy(x_s @ w.T + b, y_s)
            gw, gb = torch.autograd.grad(l_in, [w, b], create_graph=True)
            w = w - 0.1 * gw
            b = b - 0.1 * gb
        l_s = torch.nn.functional.cross_entropy(x_s @ w.T + b, y_s)
        g_s = torch.autograd.grad(l_s, [w, b], create_graph=True)
        wd = w.detach().requires_grad_(True)
        bd = b.detach().requires_grad_(True)
        l_r = torch.nn.functional.cross_entropy(x_t @ wd.T + bd, y_t)
        g_r = torch.autograd.grad(l_r, [wd, bd], create_graph=False)
        loss = sum(((g1 - g2.detach()) ** 2).sum() for g1, g2 in zip(g_s, g_r, strict=True))
        opt.zero_grad()
        loss.backward()
        opt.step()
    x_d = x_s.detach().numpy()
    y_d = np.tile([0, 1], k // 2)
    cut = n // 2
    acc_full = LogisticRegression(max_iter=300).fit(x[:cut], y[:cut]).score(x[cut:], y[cut:])
    acc_dist = LogisticRegression(max_iter=300).fit(x_d, y_d).score(x[cut:], y[cut:])
    idx = rng.choice(cut, k, replace=False)
    acc_rand = LogisticRegression(max_iter=300).fit(x[idx], y[idx]).score(x[cut:], y[cut:])
    return {
        "synthetic_dd_acc": float(acc_dist),
        "synthetic_dd_full_acc": float(acc_full),
        "synthetic_dd_random_acc": float(acc_rand),
        "synthetic_dd_recovery": float((acc_dist - acc_rand) / max(acc_full - acc_rand, 1e-9)),
        "synthetic_torch_available": 1.0,
    }
