"""BYOL bootstrap-your-own-latent (Grill et al. 2020).

Online encoder predicts an EMA target encoder's representation of the
second view — no negatives. SYNTHETIC niche: representations linear-probe
the latent class better than raw features under scale/translation views.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._ssl_synth import linear_probe_acc, make_views, synth_ssl

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("byol needs the torch `nn` extra") from exc


def bench_byol(
    seed: int = 51,
    n_train: int = 400,
    n_probe: int = 200,
    iters: int = 400,
    d_h: int = 32,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_ssl(n_train + n_probe, rng)
    xtr, ytr = x[:n_train], y[:n_train]
    xp, yp = x[n_train:], y[n_train:]
    on = torch.nn.Sequential(torch.nn.Linear(16, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h))
    tg = torch.nn.Sequential(torch.nn.Linear(16, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h))
    pred = torch.nn.Linear(d_h, d_h)
    for po, pt in zip(on.parameters(), tg.parameters(), strict=True):
        pt.data.copy_(po.data)
    for pt in tg.parameters():
        pt.requires_grad_(False)
    opt = torch.optim.Adam(list(on.parameters()) + list(pred.parameters()), lr=3e-3)
    tau = 0.99
    for _i in range(iters):
        v1, v2 = make_views(xtr, rng)
        z1 = on(torch.tensor(v1).float())
        p1 = pred(z1)
        with torch.no_grad():
            z2 = tg(torch.tensor(v2).float())
        loss = 2 - 2 * torch.nn.functional.cosine_similarity(p1, z2, -1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for po, pt in zip(on.parameters(), tg.parameters(), strict=True):
                pt.data.mul_(tau).add_((1 - tau) * po.data)
    with torch.no_grad():
        rx = on(torch.tensor(x).float()).numpy()
    acc_byol = linear_probe_acc(rx[:n_train], ytr, rx[n_train:], yp)
    acc_raw = linear_probe_acc(xtr, ytr, xp, yp)
    return {
        "synthetic_byol_probe_acc": float(acc_byol),
        "synthetic_byol_raw_acc": float(acc_raw),
        "synthetic_byol_gain": float(acc_byol - acc_raw),
        "torch_available": 1.0,
    }
