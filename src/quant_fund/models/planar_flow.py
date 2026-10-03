"""Planar Flow (Rezende & Mohamed 2015) — z' = z + u h(w^T z + b) with
logdet = log|1 + u^T h'(w^Tz+b) w|. K-step flow vs Gaussian on pinwheel.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("planar_flow requires torch (pip install -e .[nn])") from exc
    return torch


def bench_planar_flow(seed: int = 2305, iters: int = 800, K: int = 8) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    us = [torch.nn.Parameter(0.1 * torch.randn(2)) for _ in range(K)]
    ws = [torch.nn.Parameter(0.1 * torch.randn(2)) for _ in range(K)]
    bs = [torch.nn.Parameter(torch.zeros(1)) for _ in range(K)]
    params = us + ws + bs
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x):
        ld = 0.0
        for k in range(K):
            z = x
            a = z @ ws[k] + bs[k]
            hz = torch.tanh(a)
            x = z + hz[:, None] * us[k]
            psi = (1 - hz * hz)[:, None] * ws[k]
            ld = ld + torch.log(torch.abs(1 + psi @ us[k]) + 1e-9)
        return x, ld

    for _ in range(iters):
        z, ld = fwd(Xt)
        nll = (0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    with torch.no_grad():
        z, ld = fwd(torch.tensor(Xte).float())
        nll_te = float((0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean())
    base = gauss_nll(Xtr, Xte)
    return {
        "synthetic_planar_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_planar_gain": base - nll_te,
        "torch_available": 1.0,
    }
