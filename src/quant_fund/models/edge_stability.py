"""Edge-of-stability — Cohen et al. (2021): with step size η, sharpness (SYNTHETIC)
rises until ≈ 2/η then hovers. Measures sharpness trajectory vs the
2/η threshold for GD on the regime task.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._td_synth import make_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("edge_stability requires torch (pip install -e .[nn])") from exc
    return torch


def _sharp(torch, net, Xt, yt) -> float:
    params = list(net.parameters())
    n = sum(p.numel() for p in params)
    v = torch.randn(n)
    v = v / v.norm()
    lam = 0.0
    for _ in range(30):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss, params, create_graph=True)])
        Hv = torch.cat(
            [x.reshape(-1) for x in torch.autograd.grad(g @ v, params, retain_graph=True)]
        ).detach()
        lam = float(v @ Hv)
        v = Hv / (Hv.norm() + 1e-12)
    return abs(lam)


def bench_edge_stability(seed: int = 2365, iters: int = 300, lr: float = 0.05) -> dict[str, float]:
    torch = _torch()
    X, y, _, _ = make_data(seed)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(8, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()[:, None]
    sharps = []
    for i in range(iters):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 50 == 49:
            sharps.append(_sharp(torch, net, Xt, yt))
    threshold = 2.0 / lr
    frac = float(np.mean([s > 0.8 * threshold for s in sharps]))
    return {
        "synthetic_eos_threshold": float(threshold),
        "synthetic_eos_mean_sharpness": float(np.mean(sharps)),
        "synthetic_eos_frac_above_80pct": frac,
        "synthetic_torch_available": 1.0,
    }
