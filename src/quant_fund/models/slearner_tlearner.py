"""S/T-learner comparison (Künzel taxonomy) — single-net with treatment (SYNTHETIC)
feature vs two arm-specific nets; PEHE of both vs truth. Diagnostic:
T-learner wins with strong confounding on this fixture.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("slearner_tlearner requires torch (pip install -e .[nn])") from exc
    return torch


def _fit(torch, X, y, iters):
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    Xt, yt = torch.tensor(X).float(), torch.tensor(y).float()
    for _ in range(iters):
        loss = ((net(Xt).squeeze(-1) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def bench_slearner_tlearner(seed: int = 1213, iters: int = 700) -> dict[str, float]:
    torch = _torch()
    X, t, y, tau, _ = cate_data(seed)
    torch.manual_seed(seed)
    # S-learner: one net on [X, t]
    D = np.concatenate([X, t[:, None]], 1)
    s = _fit(torch, D, y, iters)
    X1 = torch.tensor(np.concatenate([X, np.ones((len(X), 1))], 1)).float()
    X0 = torch.tensor(np.concatenate([X, np.zeros((len(X), 1))], 1)).float()
    # T-learner: two nets
    t0 = _fit(torch, X[t == 0], y[t == 0], iters)
    t1 = _fit(torch, X[t == 1], y[t == 1], iters)
    Xt = torch.tensor(X).float()
    with torch.no_grad():
        s_cate = s(X1).squeeze(-1).numpy() - s(X0).squeeze(-1).numpy()
        t_cate = t1(Xt).squeeze(-1).numpy() - t0(Xt).squeeze(-1).numpy()
    return {
        "synthetic_st_s_pehe": pehe(s_cate, tau),
        "synthetic_st_t_pehe": pehe(t_cate, tau),
        "synthetic_st_t_gain": pehe(s_cate, tau) - pehe(t_cate, tau),
        "synthetic_torch_available": 1.0,
    }
