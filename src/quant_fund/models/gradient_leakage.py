"""Deep Leakage from Gradients (Zhu et al. 2019) + defense.

An attacker optimizes a dummy input to match an observed gradient;
reconstruction MSE measures leakage — and large batches / DP noise
suppress it.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import make_mlp, split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gradient_leakage requires torch (pip install -e .[nn])") from exc
    return torch


def _grad_of(torch, net, x, y):
    net.zero_grad()
    loss = torch.nn.functional.cross_entropy(net(x), y)
    g = torch.autograd.grad(loss, net.parameters())
    return [t.detach().clone() for t in g]


def _attack(torch, net, target_g, y, dim=8, steps=300, seed=0):
    torch.manual_seed(seed)
    dummy = torch.randn(1, dim, requires_grad=True)
    opt = torch.optim.Adam([dummy], lr=0.1)
    for _i in range(steps):
        net.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(dummy), y)
        g = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        dl = sum(((g[k] - target_g[k]) ** 2).sum() for k in range(len(g)))
        opt.zero_grad()
        dl.backward()
        opt.step()
    return dummy.detach()


def bench_gradient_leakage(
    seed: int = 443,
    n: int = 200,
    steps: int = 200,
) -> dict[str, float]:
    torch = _torch()
    x_tr, y_tr, x_te, y_te = split(seed, n)
    torch.manual_seed(seed)
    net = make_mlp(torch)
    # victim's single-sample gradient
    x_v = torch.tensor(x_te[:1]).float()
    y_v = torch.tensor(y_te[:1])
    g_v = _grad_of(torch, net, x_v, y_v)
    recon1 = _attack(torch, net, g_v, y_v, steps=steps, seed=seed)
    mse_single = float(((recon1 - x_v) ** 2).mean())
    # batch-of-16 gradient → harder to invert the true sample
    x_b = torch.tensor(x_te[:16]).float()
    y_b = torch.tensor(y_te[:16])
    g_b = _grad_of(torch, net, x_b, y_b)
    recon_b = _attack(torch, net, g_b, torch.tensor(y_te[:1]), steps=steps, seed=seed + 1)
    mse_batch = float(((recon_b - x_v) ** 2).mean())
    return {
        "synthetic_dlg_mse_single": mse_single,
        "synthetic_dlg_mse_batch": mse_batch,
        "synthetic_dlg_log_suppression": float(np.log10(mse_batch / max(mse_single, 1e-12))),
        "synthetic_torch_available": 1.0,
    }
