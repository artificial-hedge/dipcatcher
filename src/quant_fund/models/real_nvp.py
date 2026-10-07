"""RealNVP (Dinh et al. 2017) — affine coupling layers:
x_a → s,t nets conditioned on x_a transform x_b; logdet = Σ s.
Held-out NLL vs Gaussian on the pinwheel fixture.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("real_nvp requires torch (pip install -e .[nn])") from exc
    return torch


def _mlp(torch, d, hid=32):
    return torch.nn.Sequential(
        torch.nn.Linear(d, hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, 2),
    )


def bench_real_nvp(seed: int = 2281, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    blocks = 4
    st_nets = [_mlp(torch, 1) for _ in range(blocks)]
    params = [p for m in st_nets for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x, nets):
        ld = 0.0
        for m in nets:
            xa, xb = x[:, :1], x[:, 1:]
            st = m(xa)
            s = torch.tanh(st[:, :1])
            xb = xb * torch.exp(s) + st[:, 1:]
            ld = ld + s.squeeze(-1)
            x = torch.cat([xb, xa], -1)  # swap
        return x, ld

    for _ in range(iters):
        z, ld = fwd(Xt, st_nets)
        nll = (0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    with torch.no_grad():
        z, ld = fwd(torch.tensor(Xte).float(), st_nets)
        nll_te = float((0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean())
    base = gauss_nll(Xtr, Xte)
    return {
        "synthetic_realnvp_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_realnvp_gain": base - nll_te,
        "synthetic_torch_available": 1.0,
    }
