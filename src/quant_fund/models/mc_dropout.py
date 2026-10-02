"""MC-dropout (Gal & Ghahramani 2016) — dropout active at inference;
predictive mean/var over T stochastic forward passes. NLL + OOD gap.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mc_dropout requires torch (pip install -e .[nn])") from exc
    return torch


def bench_mc_dropout(
    seed: int = 791, iters: int = 300, T: int = 30, p: float = 0.2
) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te, x_ood = bdl_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()
    Xt = torch.tensor(x_te).float()
    Xo = torch.tensor(x_ood).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Dropout(p), torch.nn.Linear(32, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _i in range(iters):
        net.train()
        loss = ((net(X).squeeze(-1) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.train()  # keep dropout on
    with torch.no_grad():
        P = np.stack([net(Xt).squeeze(-1).numpy() for _ in range(T)])
        Po = np.stack([net(Xo).squeeze(-1).numpy() for _ in range(T)])
    mu, var = P.mean(0), P.var(0)
    return {
        "synthetic_mcdo_nll": nll_gauss(y_te, mu, var),
        "synthetic_mcdo_cov95": coverage(y_te, mu, np.sqrt(var)),
        "synthetic_mcdo_ood_gap": float(Po.var(0).mean() / (var.mean() + 1e-9)),
        "torch_available": 1.0,
    }
