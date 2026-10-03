"""CFRNet-style IPM-balanced representation (Shalit et al. 2017).

Trunk trained to predict y AND minimize an MMD penalty between
treated/control representations — balance penalty vs unpenalized
representation on PEHE.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._causal_synth import synth_observational


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("causal_rep requires torch (pip install -e .[nn])") from exc
    return torch


def _mmd(torch, z, t, kernel=0.5):
    z1 = z[t > 0.5]
    z0 = z[t < 0.5]
    if len(z1) == 0 or len(z0) == 0:
        return torch.tensor(0.0)

    def k(a, b):
        return torch.exp(-((a[:, None] - b[None, :]) ** 2).sum(-1) / kernel)

    return k(z1, z1).mean() + k(z0, z0).mean() - 2 * k(z1, z0).mean()


def bench_causal_rep(
    seed: int = 577,
    n: int = 800,
    iters: int = 120,
    ipm_w: float = 1.0,
) -> dict[str, float]:
    torch = _torch()
    x, t, y, tau, _e = synth_observational(seed, n)
    half = n // 2
    x_t = torch.tensor(x[:half]).float()
    t_t = torch.tensor(t[:half]).float()
    y_t = torch.tensor(y[:half]).float()
    x_te = torch.tensor(x[half:]).float()
    tau_te = tau[half:]

    def train(ipm):
        torch.manual_seed(seed)
        trunk = torch.nn.Sequential(torch.nn.Linear(5, 24), torch.nn.ReLU())
        h0 = torch.nn.Linear(24, 1)
        h1 = torch.nn.Linear(24, 1)
        opt = torch.optim.Adam(
            list(trunk.parameters()) + list(h0.parameters()) + list(h1.parameters()), lr=0.01
        )
        for _i in range(iters):
            z = trunk(x_t)
            pred = h1(z).squeeze(-1) * t_t + h0(z).squeeze(-1) * (1 - t_t)
            loss = ((pred - y_t) ** 2).mean() + ipm * _mmd(torch, z, t_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = trunk(x_te)
            return (h1(z) - h0(z)).squeeze(-1).numpy()

    ite_ipm = train(ipm_w)
    ite_free = train(0.0)
    pehe_i = float(np.sqrt(np.mean((ite_ipm - tau_te) ** 2)))
    pehe_f = float(np.sqrt(np.mean((ite_free - tau_te) ** 2)))
    return {
        "synthetic_crep_pehe": pehe_i,
        "synthetic_crep_free_pehe": pehe_f,
        "synthetic_crep_gain": pehe_f - pehe_i,
        "torch_available": 1.0,
    }
