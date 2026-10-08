"""Anomaly-Transformer-lite (Xu et al. 2022) (SYNTHETIC).

Two-branch attention: prior (fixed Gaussian kernel over positions) vs
series (learned attention); anomaly score = KL discrepancy between the
two association distributions — vs AE recon on windows.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._anom_synth import anom_series, auc, windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("anom_transformer requires torch (pip install -e .[nn])") from exc
    return torch


def bench_anom_transformer(
    seed: int = 627,
    w: int = 16,
    iters: int = 100,
    h: int = 16,
) -> dict[str, float]:
    torch = _torch()
    x, y = anom_series(seed=seed)
    wins, ctr = windows(x, w)
    yw = y[ctr]
    normal = wins[yw == 0]
    n_tr = int(0.7 * len(normal))
    tr = torch.tensor(normal[:n_tr]).float()
    all_w = torch.tensor(wins).float()
    torch.manual_seed(seed)
    proj = torch.nn.Linear(3, h)
    wq = torch.nn.Linear(h, h)
    wk = torch.nn.Linear(h, h)
    wv = torch.nn.Linear(h, h)
    out = torch.nn.Linear(h, 3)
    # fixed prior kernel: Gaussian in position
    pos = torch.arange(w).float()
    sig = torch.nn.Parameter(torch.tensor(1.0))
    params = (
        list(proj.parameters())
        + list(wq.parameters())
        + list(wk.parameters())
        + list(wv.parameters())
        + list(out.parameters())
        + [sig]
    )
    opt = torch.optim.Adam(params, lr=0.005)

    def branches(wx):
        z = torch.tanh(proj(wx))
        q, k, v = wq(z), wk(z), wv(z)
        s = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(h), -1)
        prior = torch.softmax(
            -0.5 * ((pos[None, :, None] - pos[None, None, :]) / sig.clamp_min(0.1)) ** 2, -1
        )
        recon = out(s @ v)
        return s, prior.expand(len(wx), -1, -1), recon

    for _i in range(iters):
        s, p, recon = branches(tr)
        recon_l = ((recon - tr) ** 2).mean()
        # maximize KL(p||s) under frozen s, minimize recon — minimax via sign flip
        kl = (p * (p.clamp_min(1e-9).log() - s.clamp_min(1e-9).log())).sum(-1).mean()
        loss = recon_l - 0.05 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s, p, recon = branches(all_w)
        kl_s = (s * (s.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).sum(-1).mean(-1)
        rec = ((recon - all_w) ** 2).mean(-1).mean(-1)
        sc = (rec * (1.0 / (kl_s / kl_s.mean().clamp_min(1e-6)).clamp_min(1e-3))).numpy()
    auc_t = auc(sc, yw)
    with torch.no_grad():
        sc_ae = ((recon - all_w) ** 2).mean(-1).mean(-1).numpy()
    auc_ae = auc(sc_ae, yw)
    return {
        "synthetic_at_auc": auc_t,
        "synthetic_at_ae_auc": auc_ae,
        "synthetic_at_gain": auc_t - auc_ae,
        "synthetic_torch_available": 1.0,
    }
