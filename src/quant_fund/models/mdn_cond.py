"""Mixture Density Network (Bishop 1994) — K-component Gaussian mixture
head on the multimodal fixture; test log-likelihood vs single-Gaussian
baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mdn_cond requires torch (pip install -e .[nn])") from exc
    return torch


def bench_mdn_cond(seed: int = 823, iters: int = 400, K: int = 3) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te = cd_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()[:, None]
    Xt = torch.tensor(x_te).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3 * K))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _i in range(iters):
        out = net(X)
        pi = torch.softmax(out[:, :K], 1)
        mu = out[:, K : 2 * K]
        sd = torch.exp(out[:, 2 * K :].clamp(-4, 3)).clamp_min(1e-3)
        comp = -0.5 * ((Y - mu) / sd) ** 2 - torch.log(sd) - 0.5 * np.log(2 * np.pi)
        ll = torch.logsumexp(torch.log(pi) + comp, 1).mean()
        loss = -ll
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = net(Xt)
        pi = torch.softmax(out[:, :K], 1)
        mu = out[:, K : 2 * K]
        sd = torch.exp(out[:, 2 * K :])
        comp = (
            -0.5 * ((torch.tensor(y_te).float()[:, None] - mu) / sd) ** 2
            - torch.log(sd)
            - 0.5 * np.log(2 * np.pi)
        )
        ll_te = torch.logsumexp(torch.log(pi) + comp, 1).numpy()
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_mdn_test_ll": float(ll_te.mean()),
        "synthetic_mdn_gauss_ll": float(ll_gauss.mean()),
        "synthetic_mdn_ll_gain": float(ll_te.mean() - ll_gauss.mean()),
        "synthetic_torch_available": 1.0,
    }
