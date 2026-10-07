"""CEVAE-lite — latent-confounder VAE for causal effect (Louizos 2017).

Infer z|x,t,y via encoder; decoder p(y|z,t); ITE via z-marginalized
counterfactual vs confounded naive ITE.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._causal_synth import synth_observational


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cevae_latent requires torch (pip install -e .[nn])") from exc
    return torch


def bench_cevae_latent(
    seed: int = 571,
    n: int = 800,
    iters: int = 150,
    d: int = 8,
) -> dict[str, float]:
    torch = _torch()
    x, t, y, tau, _e = synth_observational(seed, n)
    half = n // 2
    x_t = torch.tensor(x[:half]).float()
    t_t = torch.tensor(t[:half]).float()
    y_t = torch.tensor(y[:half]).float()
    x_te = torch.tensor(x[half:]).float()
    tau_te = tau[half:]
    torch.manual_seed(seed)
    enc = torch.nn.Sequential(torch.nn.Linear(6, 24), torch.nn.ReLU())
    mu_h = torch.nn.Linear(24, d)
    lv_h = torch.nn.Linear(24, d)
    dec = torch.nn.Sequential(torch.nn.Linear(d + 1, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    params = (
        list(enc.parameters())
        + list(mu_h.parameters())
        + list(lv_h.parameters())
        + list(dec.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.01)
    for _i in range(iters):
        h = enc(torch.cat([x_t, t_t.unsqueeze(1)], 1))
        mu = mu_h(h)
        lv = lv_h(h).clamp(-4, 2)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        pred = dec(torch.cat([z, t_t.unsqueeze(1)], 1)).squeeze(-1)
        rec = ((pred - y_t) ** 2).mean()
        kl = -0.5 * (1 + lv - mu**2 - lv.exp()).mean()
        loss = rec + 0.1 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = enc(torch.cat([x_te, torch.zeros(len(x_te), 1)], 1))
        z = mu_h(h)
        y1 = dec(torch.cat([z, torch.ones(len(x_te), 1)], 1)).squeeze(-1)
        y0 = dec(torch.cat([z, torch.zeros(len(x_te), 1)], 1)).squeeze(-1)
        ite_c = (y1 - y0).numpy()
    pehe_c = float(np.sqrt(np.mean((ite_c - tau_te) ** 2)))
    # naive: difference of raw means by arm
    ate_naive = float(y[half:][t[half:] == 1].mean() - y[half:][t[half:] == 0].mean())
    return {
        "synthetic_cevae_pehe": pehe_c,
        "synthetic_cevae_ate_err": abs(float(ite_c.mean()) - float(tau_te.mean())),
        "synthetic_cevae_naive_ate_err": abs(ate_naive - float(tau_te.mean())),
        "synthetic_torch_available": 1.0,
    }
