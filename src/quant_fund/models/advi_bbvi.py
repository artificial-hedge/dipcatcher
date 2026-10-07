"""ADVI/BBVI (Kucukelbir et al. 2017) — mean-field Gaussian VI fit by
reparameterized ELBO SGD. Posterior mean vs MCMC oracle + test log-loss.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._vi_synth import mcmc_oracle, test_logloss, vi_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("advi_bbvi requires torch (pip install -e .[nn])") from exc
    return torch


def bench_advi_bbvi(seed: int = 701, iters: int = 2500) -> dict[str, float]:
    torch = _torch()
    X, y, Xt, yt, _wt = vi_data(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    mu = torch.zeros(d, requires_grad=True)
    rho = torch.full((d,), -1.0, requires_grad=True)
    opt = torch.optim.Adam([mu, rho], lr=0.03)
    Xt_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    for _ in range(iters):
        eps = torch.randn(d)
        w = mu + torch.nn.functional.softplus(rho) * eps
        eta = torch.clamp(Xt_t @ w, -30, 30)
        ll = (
            y_t * (-torch.log1p(torch.exp(-eta)))
            + (1 - y_t) * (-eta - torch.log1p(torch.exp(-eta)))
        ).sum()
        ent = torch.log(torch.nn.functional.softplus(rho)).sum()
        prior = -0.5 * (w @ w) / 9.0
        loss = -(ll + prior + ent)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_hat = mu.detach().numpy()
    w_m, _s_m = mcmc_oracle(X, y, int(rng.integers(1 << 30)))
    base = test_logloss(np.zeros(d), Xt, yt)
    return {
        "synthetic_advi_test_logloss": test_logloss(w_hat, Xt, yt),
        "synthetic_advi_baseline_logloss": base,
        "synthetic_advi_logloss_gain": base - test_logloss(w_hat, Xt, yt),
        "synthetic_advi_mean_dev": float(
            np.linalg.norm(w_hat - w_m) / max(np.linalg.norm(w_m), 1e-9)
        ),
        "synthetic_torch_available": 1.0,
    }
