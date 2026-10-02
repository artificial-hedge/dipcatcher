"""Interval bound propagation for MLP verification.

Each linear layer's output box propagates [l,u] via the ± split of |W|;
ReLU clamps to [max(l,0), max(u,0)]. Certified accuracy = fraction of
test points whose true-class logit lower bound stays positive vs the
Monte-Carlo falsification rate at the same ε.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cert_synth import synth_cls_2d

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("ibp_bounds needs the torch `nn` extra") from exc


def bench_ibp_bounds(
    seed: int = 103,
    n_train: int = 400,
    n_test: int = 150,
    eps: float = 0.35,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_cls_2d(n_train, rng)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    xte, yte = synth_cls_2d(n_test, np.random.default_rng(seed + 1))
    xt = torch.tensor(xte).float()
    lo, hi = xt - eps, xt + eps
    w1, b1 = net[0].weight, net[0].bias
    w2, b2 = net[2].weight, net[2].bias
    c1 = (lo + hi) / 2 @ w1.T + b1
    r1 = (hi - lo) / 2 @ w1.abs().T
    lo1, hi1 = c1 - r1, c1 + r1
    lo1r, hi1r = torch.clamp(lo1, min=0), torch.clamp(hi1, min=0)
    c2 = (lo1r + hi1r) / 2 @ w2.T + b2
    r2 = (hi1r - lo1r) / 2 @ w2.abs().T
    lo2, hi2 = c2 - r2, c2 + r2
    margin = lo2 - hi2.flip(-1)
    yidx = torch.arange(xte.shape[0])
    certified = float((margin[yidx, torch.tensor(yte)] > 0).float().mean())
    with torch.no_grad():
        pred = net(xt).argmax(-1).numpy()
    acc = float((pred == yte).mean())
    rng2 = np.random.default_rng(seed + 2)
    adv = torch.tensor(xte + rng2.uniform(-eps, eps, xte.shape)).float()
    with torch.no_grad():
        pred_adv = net(adv).argmax(-1).numpy()
    frac_bad = float(((pred_adv != yte) & (pred == yte)).mean())
    return {
        "synthetic_ibp_certified_acc": certified,
        "synthetic_ibp_clean_acc": acc,
        "synthetic_ibp_mc_falsified": frac_bad,
        "synthetic_ibp_margin_mean": float(margin[yidx, torch.tensor(yte)].mean().item()),
        "torch_available": 1.0,
    }
