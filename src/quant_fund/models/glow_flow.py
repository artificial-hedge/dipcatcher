"""Glow (Kingma & Dhariwal 2018) — actnorm + invertible 1×1 + affine
coupling. 1×1 conv in 2-D = full invertible linear mixing with logdet.
Held-out NLL vs Gaussian on pinwheel.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("glow_flow requires torch (pip install -e .[nn])") from exc
    return torch


def bench_glow_flow(seed: int = 2287, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    blocks = 4
    Ws = [torch.nn.Parameter(torch.eye(2) + 0.01 * torch.randn(2, 2)) for _ in range(blocks)]
    ss = [torch.nn.Parameter(torch.zeros(2)) for _ in range(blocks)]
    bs = [torch.nn.Parameter(torch.zeros(2)) for _ in range(blocks)]
    st_nets = [
        torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        for _ in range(blocks)
    ]
    params = Ws + ss + bs + [p for m in st_nets for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x):
        ld = 0.0
        for i in range(blocks):
            # actnorm
            x = x * torch.exp(ss[i]) + bs[i]
            ld = ld + ss[i].sum()
            # 1x1
            W = Ws[i]
            x = x @ W.t()
            ld = ld + torch.slogdet(W)[1]
            # coupling: x0 → (s,t) for x1
            xa, xb = x[:, :1], x[:, 1:]
            st = st_nets[i](xa)
            s = torch.tanh(st[:, :1])
            xb = xb * torch.exp(s) + st[:, 1:]
            ld = ld + s.squeeze(-1)
            x = torch.cat([xa, xb], -1)
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
        "synthetic_glow_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_glow_gain": base - nll_te,
        "synthetic_torch_available": 1.0,
    }
