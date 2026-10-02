"""IAF (Kingma et al. 2016) — inverse autoregressive flow: z' = z*σ(x_<i)
+ μ(x_<i); sampling-parallel (vs MAF which is density-parallel). NLL vs
Gaussian on pinwheel.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("iaf_flow requires torch (pip install -e .[nn])") from exc
    return torch


def bench_iaf_flow(seed: int = 2311, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    # IAF density: x = z*σ(z_<i ctx) + μ — evaluate via autoregression on x
    nets = [
        torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        for _ in range(3)
    ]
    params = [p for m in nets for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x):
        # invert autoregressively: z0 = x0; z1 = (x1 - μ(z0)) / σ(z0)
        ld = 0.0
        for m in nets:
            z0 = x[:, :1]
            st = m(z0)
            s = torch.tanh(st[:, :1])
            z1 = (x[:, 1:] - st[:, 1:]) * torch.exp(-s)
            ld = ld - s.squeeze(-1)
            x = torch.cat([z1, z0], -1)  # permute for next layer
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
        "synthetic_iaf_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_iaf_gain": base - nll_te,
        "torch_available": 1.0,
    }
