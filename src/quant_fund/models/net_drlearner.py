"""DR-learner (Kennedy 2020) — doubly-robust pseudo-outcome regressed (SYNTHETIC)
by a small net: phi = (m1−m0) + (t−e)/(e(1−e))·(y−m_t). PEHE vs plugin.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("net_drlearner requires torch (pip install -e .[nn])") from exc
    return torch


def bench_net_drlearner(seed: int = 1229, iters: int = 800) -> dict[str, float]:
    torch = _torch()
    X, t, y, tau, _ = cate_data(seed)
    torch.manual_seed(seed)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()
    tt = torch.tensor(t).float()
    m = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1] + 1, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    e = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    opt = torch.optim.Adam(list(m.parameters()) + list(e.parameters()), lr=0.01)
    for _ in range(iters):
        lm = ((m(torch.cat([Xt, tt[:, None]], 1)).squeeze(-1) - yt) ** 2).mean()
        le = torch.nn.functional.binary_cross_entropy_with_logits(e(Xt).squeeze(-1), tt)
        opt.zero_grad()
        (lm + le).backward()
        opt.step()
    with torch.no_grad():
        e0 = torch.sigmoid(e(Xt)).squeeze(-1).numpy()
        m1 = m(torch.cat([Xt, torch.ones(len(X), 1)], 1)).squeeze(-1).numpy()
        m0 = m(torch.cat([Xt, torch.zeros(len(X), 1)], 1)).squeeze(-1).numpy()
        ec = np.clip(e0, 0.05, 0.95)
        mt = np.where(t == 1, m1, m0)
        pseudo = (m1 - m0) + (t - ec) / (ec * (1 - ec)) * (y - mt)
    tau_net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    opt2 = torch.optim.Adam(tau_net.parameters(), lr=0.01)
    pt = torch.tensor(pseudo).float()
    for _ in range(iters):
        loss = ((tau_net(Xt).squeeze(-1) - pt) ** 2).mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        cate = tau_net(Xt).squeeze(-1).numpy()
    return {
        "synthetic_ndr_pehe": pehe(cate, tau),
        "synthetic_ndr_plugin_pehe": pehe(m1 - m0, tau),
        "synthetic_ndr_pehe_gain": pehe(m1 - m0, tau) - pehe(cate, tau),
        "synthetic_torch_available": 1.0,
    }
