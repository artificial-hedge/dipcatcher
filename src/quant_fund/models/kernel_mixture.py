"""Kernel Mixture Network (Ambrogioni et al. 2017, Rothfuss-style) — (SYNTHETIC)
conditioned Gaussian-kernel mixture: net emits (pi_j, mu_j) for fixed
bandwidth grid → conditional density. Test LL vs Gaussian.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("kernel_mixture requires torch (pip install -e .[nn])") from exc
    return torch


def bench_kernel_mixture(
    seed: int = 841, iters: int = 400, K: int = 8, bw: float = 0.3
) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te = cd_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()[:, None]
    Xt = torch.tensor(x_te).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(3, 48), torch.nn.Tanh(), torch.nn.Linear(48, 2 * K))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _i in range(iters):
        out = net(X)
        pi = torch.softmax(out[:, :K], 1)
        mu = out[:, K:]
        kern = -0.5 * ((Y - mu) / bw) ** 2 - np.log(bw) - 0.5 * np.log(2 * np.pi)
        ll = torch.logsumexp(torch.log(pi) + kern, 1).mean()
        loss = -ll
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = net(Xt)
        pi = torch.softmax(out[:, :K], 1)
        mu = out[:, K:]
        kern = (
            -0.5 * ((torch.tensor(y_te).float()[:, None] - mu) / bw) ** 2
            - np.log(bw)
            - 0.5 * np.log(2 * np.pi)
        )
        ll_te = torch.logsumexp(torch.log(pi) + kern, 1).numpy()
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_kmn_test_ll": float(ll_te.mean()),
        "synthetic_kmn_gauss_ll": float(ll_gauss.mean()),
        "synthetic_kmn_ll_gain": float(ll_te.mean() - ll_gauss.mean()),
        "synthetic_torch_available": 1.0,
    }
