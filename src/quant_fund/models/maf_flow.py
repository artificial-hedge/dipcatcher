"""MAF (Papamakarios et al. 2017) — masked autoregressive flow:
each dim's scale/shift conditioned on previous dims (MADE-style masks).
NLL vs Gaussian on pinwheel.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("maf_flow requires torch (pip install -e .[nn])") from exc
    return torch


def bench_maf_flow(seed: int = 2299, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    # 2-D: dim1 → (s2, t2); dim0 standard normal. 3 autoregressive stacks.
    nets = [
        torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        for _ in range(3)
    ]
    params = [p for m in nets for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x):
        ld = 0.0
        for m in nets:
            st = m(x[:, :1])
            s = torch.tanh(st[:, :1])
            z0 = x[:, :1]  # dim0 already base
            z1 = (x[:, 1:] - st[:, 1:]) * torch.exp(-s)
            ld = ld - s.squeeze(-1)
            x = torch.cat([z0, z1], -1)
            x = x[:, [1, 0]]  # permute
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
        "synthetic_maf_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_maf_gain": base - nll_te,
        "synthetic_torch_available": 1.0,
    }
