"""X-learner (Künzel et al. 2019) — two-stage: fit outcome models per
arm, impute counterfactual effects, re-fit on imputed CATE, combine
with propensity weights. PEHE vs S-learner.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("xlearner requires torch (pip install -e .[nn])") from exc
    return torch


def _fit_net(torch, X, y, iters: int = 700):
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


def bench_xlearner(seed: int = 1201, iters: int = 700) -> dict[str, float]:
    torch = _torch()
    X, t, y, tau, _ = cate_data(seed)
    torch.manual_seed(seed)
    m0 = _fit_net(torch, X[t == 0], y[t == 0], iters)
    m1 = _fit_net(torch, X[t == 1], y[t == 1], iters)
    Xt = torch.tensor(X).float()
    with torch.no_grad():
        d1 = y - m0(Xt).squeeze(-1).numpy()  # imputed effect on controls' scale? standard:
        d0 = m1(Xt).squeeze(-1).numpy() - y
    d1_true_fit = np.where(t == 1, d1, 0.0)
    d0_true_fit = np.where(t == 0, d0, 0.0)
    t1 = _fit_net(torch, X[t == 1], d1_true_fit[t == 1], iters // 2)
    t0 = _fit_net(torch, X[t == 0], d0_true_fit[t == 0], iters // 2)
    with torch.no_grad():
        g = np.clip(1 / (1 + np.exp(-1.5 * X[:, 0] + 0.3)), 0.05, 0.95)
        cate = g * t0(Xt).squeeze(-1).numpy() + (1 - g) * t1(Xt).squeeze(-1).numpy()
        s_cate = m1(Xt).squeeze(-1).numpy() - m0(Xt).squeeze(-1).numpy()
    return {
        "synthetic_xlearner_pehe": pehe(cate, tau),
        "synthetic_xlearner_s_pehe": pehe(s_cate, tau),
        "synthetic_xlearner_pehe_gain": pehe(s_cate, tau) - pehe(cate, tau),
        "synthetic_torch_available": 1.0,
    }
