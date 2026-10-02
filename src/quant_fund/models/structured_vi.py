"""Structured VI — full-rank Cholesky Gaussian posterior vs mean-field:
captures posterior covariance the mean-field family misses. Covariance
Frobenius deviation vs MCMC oracle, both families.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._vi_synth import mcmc_oracle, test_logloss, vi_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("structured_vi requires torch (pip install -e .[nn])") from exc
    return torch


def bench_structured_vi(seed: int = 733, iters: int = 2200) -> dict[str, float]:
    torch = _torch()
    X, y, Xt, yt, _wt = vi_data(seed, n=200)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Xt_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    # MCMC oracle covariance
    w_m, s_m = mcmc_oracle(X, y, int(rng.integers(1 << 30)), iters=8000)
    # mean-field
    mu_mf = torch.zeros(d, requires_grad=True)
    rho_mf = torch.full((d,), -1.0, requires_grad=True)
    # full-rank
    mu_fr = torch.zeros(d, requires_grad=True)
    L_raw = (torch.eye(d) * 0.4).requires_grad_()
    opt = torch.optim.Adam(
        [{"params": [mu_mf, rho_mf]}, {"params": [mu_fr, L_raw], "lr": 0.002}], lr=0.01
    )
    S = 8
    for _ in range(iters):
        sig = torch.nn.functional.softplus(rho_mf)
        L = (
            torch.tril(L_raw)
            - torch.diag(torch.diagonal(L_raw))
            + torch.diag(torch.nn.functional.softplus(torch.diagonal(L_raw)) + 0.05)
        )
        eps = torch.randn(S, d)
        lq = torch.tensor(0.0)
        for fam in ("mf", "fr"):
            if fam == "mf":
                w = mu_mf + sig * eps
                lqs = (
                    -0.5 * ((w - mu_mf) / sig) ** 2
                    - torch.log(sig)
                    - 0.5 * float(np.log(2 * np.pi))
                ).sum(1)
            else:
                w = mu_fr + eps @ L.T
                lqs = (
                    -0.5 * ((eps * eps).sum(1))
                    - torch.log(torch.diag(L)).sum()
                    - 0.5 * d * float(np.log(2 * np.pi))
                )
            eta = torch.clamp(Xt_t @ w.T, -30, 30)
            ll = (
                y_t[:, None] * (-torch.log1p(torch.exp(-eta)))
                + (1 - y_t[:, None]) * (-eta - torch.log1p(torch.exp(-eta)))
            ).sum(0)
            lq = lq + ((lqs - ll + 0.5 * (w * w).sum(1) / 9.0).mean())
        opt.zero_grad()
        lq.backward()
        opt.step()
    with torch.no_grad():
        cov_fr = (L_raw @ L_raw.T).numpy()
    # crude MCMC covariance
    rng2 = np.random.default_rng(seed + 1)
    w = np.zeros(d)
    acc = []
    X_, y_ = X, y
    from quant_fund.models._vi_synth import logpost

    lp = logpost(w, X_, y_)
    for _ in range(8000):
        wp = w + 0.15 * rng2.standard_normal(d)
        lp2 = logpost(wp, X_, y_)
        if np.log(rng2.random()) < lp2 - lp:
            w, lp = wp, lp2
        acc.append(w.copy())
    cov_m = np.cov(np.asarray(acc[4000:]).T)
    cov_mf = np.diag((np.log1p(np.exp(rho_mf.detach().numpy()))) ** 2)
    dev_mf = float(np.linalg.norm(cov_mf - cov_m) / max(np.linalg.norm(cov_m), 1e-9))
    dev_fr = float(np.linalg.norm(cov_fr - cov_m) / max(np.linalg.norm(cov_m), 1e-9))
    base = test_logloss(np.zeros(d), Xt, yt)
    return {
        "synthetic_svi_full_test_logloss": test_logloss(mu_fr.detach().numpy(), Xt, yt),
        "synthetic_svi_logloss_gain": base - test_logloss(mu_fr.detach().numpy(), Xt, yt),
        "synthetic_svi_cov_dev_mf": dev_mf,
        "synthetic_svi_cov_dev_fr": dev_fr,
        "synthetic_svi_cov_gain": dev_mf - dev_fr,
        "synthetic_svi_oracle_sd": float(np.linalg.norm(s_m)),
        "torch_available": 1.0,
    }
