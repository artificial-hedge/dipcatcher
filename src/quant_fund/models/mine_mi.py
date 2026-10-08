"""MINE (Belghazi et al. 2018): neural MI lower bound via DV estimator (SYNTHETIC)
E_joint[T] - log E_marg[e^T]. Torch-gated. Trained on dependent vs
independent Gaussians; gap vs true MI 0.34.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._it_synth import dep_data, true_mi_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mine_mi requires torch (pip install -e .[nn])") from exc
    return torch


def _mine(x: np.ndarray, y: np.ndarray, seed: int, iters: int = 600) -> float:
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
        Tm = torch.logsumexp(net(Xm).squeeze(-1), 0) - np.log(len(X))
        loss = -(Tj - Tm)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        perm = torch.randperm(len(X), generator=g)
        Xm = torch.cat([X[:, :1], X[perm, 1:]], -1)
        est = float(net(X).mean() - (torch.logsumexp(net(Xm).squeeze(-1), 0) - np.log(len(X))))
    return max(est, 0.0)


def bench_mine_mi(seed: int = 2887) -> dict[str, float]:
    x, y = dep_data(seed, kind="dep")  # corr ~0.97
    mi_dep = _mine(x, y, seed)
    x0, y0 = dep_data(seed + 1, kind="indep")
    mi_ind = _mine(x0, y0, seed + 1)
    true_dep = true_mi_gauss(0.9701)
    return {
        "synthetic_mine_mi_dep": float(mi_dep),
        "synthetic_mine_mi_indep": float(mi_ind),
        "synthetic_mine_true_mi": float(true_dep),
        "synthetic_mine_gap": float(abs(mi_dep - true_dep)),
        "synthetic_torch_available": 1.0,
    }
