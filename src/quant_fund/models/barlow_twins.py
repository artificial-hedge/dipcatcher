"""Barlow Twins redundancy reduction (Zbontar et al. 2021).

Cross-correlation of two views driven to identity — invariance on the
diagonal, decorrelation off it. No negatives, no asymmetry tricks.
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
        raise ImportError("barlow_twins needs the torch `nn` extra") from exc


def bench_barlow_twins(
    seed: int = 53,
    n_train: int = 400,
    n_probe: int = 200,
    iters: int = 400,
    d_h: int = 32,
    lam: float = 5e-3,
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
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        c = (z1.T @ z2) / z1.shape[0]
        eye = torch.eye(d_h)
        loss = ((c - eye) ** 2 * (eye + lam * (1 - eye))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rx = net(torch.tensor(x).float()).numpy()
    acc_bt = linear_probe_acc(rx[:n_train], ytr, rx[n_train:], yp)
    acc_raw = linear_probe_acc(xtr, ytr, xp, yp)
    return {
        "synthetic_barlow_probe_acc": float(acc_bt),
        "synthetic_barlow_raw_acc": float(acc_raw),
        "synthetic_barlow_gain": float(acc_bt - acc_raw),
        "synthetic_torch_available": 1.0,
    }
