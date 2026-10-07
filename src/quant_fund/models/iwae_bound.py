"""IWAE (Burda et al. 2015) — importance-weighted autoencoder: K-sample (SYNTHETIC)
ELBO bound trained with the SNIS gradient estimator on the logistic
regression posterior. Tighter bound + posterior mean vs MCMC oracle.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._vi_synth import mcmc_oracle, test_logloss, vi_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("iwae_bound requires torch (pip install -e .[nn])") from exc
    return torch


def bench_iwae_bound(seed: int = 709, iters: int = 1500, K: int = 16) -> dict[str, float]:
    torch = _torch()
    X, y, Xt, yt, _wt = vi_data(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    mu = torch.zeros(d, requires_grad=True)
    rho = torch.full((d,), -0.5, requires_grad=True)
    opt = torch.optim.Adam([mu, rho], lr=0.03)
    Xt_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    final_bound = torch.tensor(0.0)
    for _ in range(iters):
        sig = torch.nn.functional.softplus(rho)
        eps = torch.randn(K, d)
        w = mu + sig * eps
        eta = torch.clamp(Xt_t @ w.T, -30, 30)  # n x K
        ll = (
            y_t[:, None] * (-torch.log1p(torch.exp(-eta)))
            + (1 - y_t[:, None]) * (-eta - torch.log1p(torch.exp(-eta)))
        ).sum(0)
        log_q = -0.5 * ((w - mu) / sig) ** 2 - torch.log(sig) - 0.5 * float(np.log(2 * np.pi))
        log_q = log_q.sum(1)
        log_p = ll - 0.5 * (w * w).sum(1) / 9.0
        lw = log_p - log_q
        final_bound = torch.logsumexp(lw, 0) - float(np.log(K))
        loss = -final_bound
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_hat = mu.detach().numpy()
    w_m, _s_m = mcmc_oracle(X, y, int(rng.integers(1 << 30)))
    # naive 1-sample ELBO at final params for comparison
    sig = torch.nn.functional.softplus(rho).detach()
    eps = torch.randn(256, d)
    w = mu.detach() + sig * eps
    eta = torch.clamp(Xt_t @ w.T, -30, 30)
    ll = (
        y_t[:, None] * (-torch.log1p(torch.exp(-eta)))
        + (1 - y_t[:, None]) * (-eta - torch.log1p(torch.exp(-eta)))
    ).sum(0)
    elbo1 = float(
        (
            ll
            - 0.5 * (w * w).sum(1) / 9.0
            + (torch.log(sig).sum() + 0.5 * d * (1 + float(np.log(2 * np.pi))))
        ).mean()
    )
    return {
        "synthetic_iwae_test_logloss": test_logloss(w_hat, Xt, yt),
        "synthetic_iwae_elbo1": elbo1,
        "synthetic_iwae_bound_k": float(final_bound),
        "synthetic_iwae_bound_gap": float(final_bound) - elbo1,
        "synthetic_iwae_mean_dev": float(
            np.linalg.norm(w_hat - w_m) / max(np.linalg.norm(w_m), 1e-9)
        ),
        "synthetic_torch_available": 1.0,
    }
