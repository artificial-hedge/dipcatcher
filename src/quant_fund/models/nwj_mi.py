"""NWJ / f-GAN MI estimator (Nguyen, Wainwright & Jordan 2010): (SYNTHETIC)
E_joint[T] - E_marg[e^(T-1)]. Torch-gated twin-net to MINE.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import dep_data, true_mi_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("nwj_mi requires torch (pip install -e .[nn])") from exc
    return torch


def _nwj(x: np.ndarray, y: np.ndarray, seed: int, iters: int = 800) -> float:
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    for p in net.parameters():
        if p.ndim > 1:
            torch.nn.init.xavier_uniform_(p, generator=g)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    X = torch.tensor(np.stack([x, y], -1), dtype=torch.float32)
    for _ in range(iters):
        perm = torch.randperm(len(X), generator=g)
        Xm = torch.cat([X[:, :1], X[perm, 1:]], -1)
        Tj = net(X).mean()
        Tm = torch.exp(net(Xm).squeeze(-1) - 1.0).mean()
        loss = -(Tj - Tm)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        perm = torch.randperm(len(X), generator=g)
        Xm = torch.cat([X[:, :1], X[perm, 1:]], -1)
        est = float(net(X).mean() - torch.exp(net(Xm).squeeze(-1) - 1.0).mean() + 1.0)
    return max(est, 0.0)


def bench_nwj_mi(seed: int = 2893) -> dict[str, float]:
    x, y = dep_data(seed, kind="dep")
    mi_dep = _nwj(x, y, seed)
    x0, y0 = dep_data(seed + 1, kind="indep")
    mi_ind = _nwj(x0, y0, seed + 1)
    true_dep = true_mi_gauss(0.9701)
    return {
        "synthetic_nwj_mi_dep": float(mi_dep),
        "synthetic_nwj_mi_indep": float(mi_ind),
        "synthetic_nwj_true_mi": float(true_dep),
        "synthetic_nwj_gap": float(abs(mi_dep - true_dep)),
        "synthetic_torch_available": 1.0,
    }
