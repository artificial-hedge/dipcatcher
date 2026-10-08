"""Diffusion regressor — conditional DDPM sampler on y|x (eps-prediction, (SYNTHETIC)
T=30 steps); test log-density approximated by sampling spread around
posterior mean vs Gaussian baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("diffusion_regressor requires torch (pip install -e .[nn])") from exc
    return torch


def bench_diffusion_regressor(
    seed: int = 829, iters: int = 400, T: int = 30, n_samp: int = 16
) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te = cd_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()[:, None]
    Xt = torch.tensor(x_te).float()
    betas = np.linspace(1e-3, 0.02, T)
    alphas = 1 - betas
    ab = np.cumprod(alphas)
    torch.manual_seed(seed)
    eps_net = torch.nn.Sequential(
        torch.nn.Linear(4 + 1, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(eps_net.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    for _i in range(iters):
        t = rng.integers(1, T + 1, len(X))
        ab_t = torch.tensor(ab[t - 1]).float()[:, None]
        eps = torch.randn(len(X), 1)
        y_t = torch.sqrt(ab_t) * Y + torch.sqrt(1 - ab_t) * eps
        inp = torch.cat([X, y_t, torch.tensor(t / T).float()[:, None]], 1)
        loss = ((eps_net(inp) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # ancestral sample per test point
    with torch.no_grad():
        ys = torch.randn(n_samp, len(Xt), 1)
        for tt in range(T, 0, -1):
            ab_t = ab[tt - 1]
            alpha = alphas[tt - 1]
            beta = betas[tt - 1]
            tin = torch.full((len(Xt), 1), tt / T)
            for si in range(n_samp):
                inp = torch.cat([Xt, ys[si], tin], 1)
                e = eps_net(inp)
                ys[si] = (ys[si] - (1 - alpha) / np.sqrt(1 - ab_t) * e) / np.sqrt(alpha) + (
                    np.sqrt(beta) * torch.randn(len(Xt), 1) if tt > 1 else 0
                )
        samp = ys.numpy()
    mu = samp.mean(0).squeeze(-1)
    var = samp.var(0).squeeze(-1) + 0.01
    ll_te = -0.5 * np.log(2 * np.pi * var) - (y_te - mu) ** 2 / (2 * var)
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_diffreg_test_ll": float(ll_te.mean()),
        "synthetic_diffreg_gauss_ll": float(ll_gauss.mean()),
        "synthetic_diffreg_ll_gain": float(ll_te.mean() - ll_gauss.mean()),
        "synthetic_torch_available": 1.0,
    }
