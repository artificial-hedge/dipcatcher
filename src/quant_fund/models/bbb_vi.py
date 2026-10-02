"""Bayes-by-Backprop (Blundell et al. 2015) — diagonal Gaussian posterior
over ALL weights; ELBO = E_q[NLL] - KL(q||N(0,1)); local reparam.
Predictive var = weight-uncertainty spread. NLL + OOD gap.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("bbb_vi requires torch (pip install -e .[nn])") from exc
    return torch


def bench_bbb_vi(seed: int = 797, iters: int = 200, T: int = 20) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te, x_ood = bdl_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()
    Xt = torch.tensor(x_te).float()
    Xo = torch.tensor(x_ood).float()
    torch.manual_seed(seed)
    # single-layer BBB (linear): w ~ N(mu, softplus(rho)^2)
    mu = torch.nn.Parameter(torch.randn(4) * 0.1)
    rho = torch.nn.Parameter(torch.full((4,), -3.0))
    b_mu = torch.nn.Parameter(torch.zeros(1))
    b_rho = torch.nn.Parameter(torch.full((1,), -3.0))
    opt = torch.optim.Adam([mu, rho, b_mu, b_rho], lr=0.02)
    for _i in range(iters):
        sig = torch.nn.functional.softplus(rho)
        sb = torch.nn.functional.softplus(b_rho)
        w = mu + sig * torch.randn(4)
        b = b_mu + sb * torch.randn(1)
        pred = X @ w + b
        nll = ((pred - Y) ** 2).mean() / (2 * 0.09)  # known-ish noise var 0.09
        kl = (0.5 * (mu**2 + sig**2 - 1 - torch.log(sig**2))).sum() + (
            0.5 * (b_mu**2 + sb**2 - 1 - torch.log(sb**2))
        ).sum()
        loss = nll + kl / len(Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    preds_te, preds_ood = [], []
    with torch.no_grad():
        sig = torch.nn.functional.softplus(rho)
        sb = torch.nn.functional.softplus(b_rho)
        for _ in range(T):
            w = mu + sig * torch.randn(4)
            b = b_mu + sb * torch.randn(1)
            preds_te.append((Xt @ w + b).numpy())
            preds_ood.append((Xo @ w + b).numpy())
    P = np.asarray(preds_te)
    mu_p, var = P.mean(0), P.var(0) + 0.09
    Po = np.asarray(preds_ood)
    return {
        "synthetic_bbb_nll": nll_gauss(y_te, mu_p, var),
        "synthetic_bbb_cov95": coverage(y_te, mu_p, np.sqrt(var)),
        "synthetic_bbb_ood_gap": float(Po.var(0).mean() / (P.var(0).mean() + 1e-9)),
        "torch_available": 1.0,
    }
