"""Concrete dropout (Gal et al. 2017) — learnable dropout probability per
layer via the concrete relaxation; trained jointly with weights. Reports
learned p and NLL + OOD gap from MC samples at inference.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("concrete_dropout requires torch (pip install -e .[nn])") from exc
    return torch


def bench_concrete_dropout(seed: int = 809, iters: int = 300, T: int = 30) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te, x_ood = bdl_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()
    Xt = torch.tensor(x_te).float()
    Xo = torch.tensor(x_ood).float()
    torch.manual_seed(seed)
    lin1 = torch.nn.Linear(4, 32)
    lin2 = torch.nn.Linear(32, 1)
    logit_p = torch.nn.Parameter(torch.tensor(-1.0))  # p≈0.27 init
    params = list(lin1.parameters()) + list(lin2.parameters()) + [logit_p]
    opt = torch.optim.Adam(params, lr=0.01)
    temp = 0.067
    reg = 1e-3

    def forward(xx):
        p = torch.sigmoid(logit_p)
        u = torch.rand(xx.shape[0], 32)
        conc = torch.sigmoid((torch.log(u) - torch.log(1 - u) + torch.log(p)) / temp)
        keep = conc / (1 - p).clamp_min(1e-6)
        h = torch.relu(lin1(xx)) * keep
        return lin2(h).squeeze(-1)

    for _i in range(iters):
        pred = forward(X)
        p = torch.sigmoid(logit_p)
        loss = ((pred - Y) ** 2).mean() + reg * p
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = np.stack([forward(Xt).numpy() for _ in range(T)])
        Po = np.stack([forward(Xo).numpy() for _ in range(T)])
    mu, var = P.mean(0), P.var(0)
    return {
        "synthetic_cd_nll": nll_gauss(y_te, mu, var),
        "synthetic_cd_cov95": coverage(y_te, mu, np.sqrt(var)),
        "synthetic_cd_ood_gap": float(Po.var(0).mean() / (var.mean() + 1e-9)),
        "synthetic_cd_learned_p": float(torch.sigmoid(logit_p).item()),
    }
