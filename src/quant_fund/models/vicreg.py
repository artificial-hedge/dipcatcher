"""VICReg variance-invariance-covariance regularization (Bardes et al. 2022).

Invariance term + per-dimension hinge variance + covariance penalty —
explicit collapse prevention without negatives or stop-grad tricks.
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
        raise ImportError("vicreg needs the torch `nn` extra") from exc


def bench_vicreg(
    seed: int = 57,
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
    net = torch.nn.Sequential(torch.nn.Linear(16, d_h), torch.nn.ReLU(), torch.nn.Linear(d_h, d_h))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _i in range(iters):
        v1, v2 = make_views(xtr, rng)
        z1 = net(torch.tensor(v1).float())
        z2 = net(torch.tensor(v2).float())
        sim = (z1 - z2).pow(2).mean()
        var = torch.relu(1.0 - z1.std(0)).mean() + torch.relu(1.0 - z2.std(0)).mean()
        z1c = z1 - z1.mean(0)
        z2c = z2 - z2.mean(0)
        c1 = (z1c.T @ z1c) / (z1c.shape[0] - 1)
        c2 = (z2c.T @ z2c) / (z2c.shape[0] - 1)
        off1 = (c1 * (1 - torch.eye(d_h))).pow(2).sum() / d_h
        off2 = (c2 * (1 - torch.eye(d_h))).pow(2).sum() / d_h
        loss = 25.0 * sim + 25.0 * var + (off1 + off2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rx = net(torch.tensor(x).float()).numpy()
    acc_v = linear_probe_acc(rx[:n_train], ytr, rx[n_train:], yp)
    acc_raw = linear_probe_acc(xtr, ytr, xp, yp)
    return {
        "synthetic_vicreg_probe_acc": float(acc_v),
        "synthetic_vicreg_raw_acc": float(acc_raw),
        "synthetic_vicreg_gain": float(acc_v - acc_raw),
        "synthetic_torch_available": 1.0,
    }
