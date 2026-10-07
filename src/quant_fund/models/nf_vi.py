"""Normalizing-flow VI (Rezende & Mohamed 2015) — planar flow
z' = z + u·tanh(w·z + b) applied to the posterior sample; richer than
mean-field Gaussian. Posterior std vs MCMC + test log-loss.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._vi_synth import mcmc_oracle, test_logloss, vi_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("nf_vi requires torch (pip install -e .[nn])") from exc
    return torch


def bench_nf_vi(seed: int = 719, iters: int = 2000, Kflow: int = 3) -> dict[str, float]:
    torch = _torch()
    X, y, Xt, yt, _wt = vi_data(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    mu = torch.zeros(d, requires_grad=True)
    rho = torch.full((d,), -1.0, requires_grad=True)
    U = torch.zeros(Kflow, d, requires_grad=True)
    W = torch.zeros(Kflow, d, requires_grad=True)
    B = torch.zeros(Kflow, requires_grad=True)
    opt = torch.optim.Adam([mu, rho, U, W, B], lr=0.02)
    Xt_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    S = 8
    for _ in range(iters):
        sig = torch.nn.functional.softplus(rho)
        eps = torch.randn(S, d)
        z0 = mu + sig * eps
        log_q0 = (
            -0.5 * ((z0 - mu) / sig) ** 2 - torch.log(sig) - 0.5 * float(np.log(2 * np.pi))
        ).sum(1)
        z, logdet = z0, torch.zeros(S)
        for k in range(Kflow):
            h = torch.tanh(z @ W[k] + B[k])
            psi = (1 - h**2)[:, None] * W[k]
            z = z + U[k] * h[:, None]
            logdet = logdet + torch.log((1 + psi @ U[k]).abs().clamp_min(1e-6))
        log_q = log_q0 - logdet
        eta = torch.clamp(Xt_t @ z.T, -30, 30)
        ll = (
            y_t[:, None] * (-torch.log1p(torch.exp(-eta)))
            + (1 - y_t[:, None]) * (-eta - torch.log1p(torch.exp(-eta)))
        ).sum(0)
        log_p = ll - 0.5 * (z * z).sum(1) / 9.0
        loss = (log_q - log_p).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        sig = torch.nn.functional.softplus(rho)
        eps = torch.randn(512, d)
        z = mu + sig * eps
        for k in range(Kflow):
            h = torch.tanh(z @ W[k] + B[k])
            z = z + U[k] * h[:, None]
        w_hat = z.mean(0).numpy()
        sd_hat = z.std(0).numpy()
    w_m, s_m = mcmc_oracle(X, y, int(rng.integers(1 << 30)))
    base = test_logloss(np.zeros(d), Xt, yt)
    return {
        "synthetic_nfv_test_logloss": test_logloss(w_hat, Xt, yt),
        "synthetic_nfv_logloss_gain": base - test_logloss(w_hat, Xt, yt),
        "synthetic_nfv_sd_dev": float(
            np.linalg.norm(sd_hat - s_m) / max(np.linalg.norm(s_m), 1e-9)
        ),
        "synthetic_nfv_mean_dev": float(
            np.linalg.norm(w_hat - w_m) / max(np.linalg.norm(w_m), 1e-9)
        ),
        "synthetic_torch_available": 1.0,
    }
