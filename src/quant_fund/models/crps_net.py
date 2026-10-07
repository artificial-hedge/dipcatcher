"""Direct-CRPS net (Berrisch & Ziel 2021) — predict full conditional via
approximated CRPS of a sample-path discretization (closed-form CRPS for
Gaussian + mixture on a fixed y-grid). Test NLL proxy via kernel density
of the fitted distribution vs Gaussian.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("crps_net requires torch (pip install -e .[nn])") from exc
    return torch


def bench_crps_net(seed: int = 839, iters: int = 400, K: int = 64) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te = cd_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()[:, None]
    Xt = torch.tensor(x_te).float()
    grid = torch.linspace(float(y.min()) - 1, float(y.max()) + 1, K)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(3, 48), torch.nn.Tanh(), torch.nn.Linear(48, K))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    dz = float(grid[1] - grid[0])
    for _i in range(iters):
        logits = net(X)
        pdf = torch.softmax(logits, 1) / dz
        cdf = torch.cumsum(pdf, 1) * dz
        # CRPS: ∫(F(z) − 1{z≥y})² dz
        ind = (grid[None, :] >= Y).float()
        crps = ((cdf - ind) ** 2).sum(1) * dz
        loss = crps.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = net(Xt)
        pm = torch.softmax(logits, 1).numpy()  # bin masses
        g = grid.numpy()
        # smooth mass → Gaussian-mixture density at bw=dz
        kern = np.exp(-0.5 * ((y_te[:, None] - g[None, :]) / dz) ** 2) / (dz * np.sqrt(2 * np.pi))
        dens = (pm * kern).sum(1)
    ll_te = np.log(dens + 1e-9)
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_crps_test_ll": float(ll_te.mean()),
        "synthetic_crps_gauss_ll": float(ll_gauss.mean()),
        "synthetic_crps_ll_gain": float(ll_te.mean() - ll_gauss.mean()),
        "synthetic_torch_available": 1.0,
    }
