"""R-learner (Nie & Wager 2021) — Robinson residual-on-residual:
estimate m(x)=E[Y|X], e(x)=P(T=1|X), then fit tau(x) minimizing
((y−m) − tau·(t−e))². PEHE vs direct diff regressor.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._cate_synth import cate_data, pehe


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("rlearner requires torch (pip install -e .[nn])") from exc
    return torch


def bench_rlearner(seed: int = 1207, iters: int = 900) -> dict[str, float]:
    torch = _torch()
    X, t, y, tau, _ = cate_data(seed)
    torch.manual_seed(seed)
    Xt = torch.tensor(X).float()
    yt = torch.tensor(y).float()
    tt = torch.tensor(t).float()
    m = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    e = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    opt = torch.optim.Adam(list(m.parameters()) + list(e.parameters()), lr=0.01)
    for _ in range(iters):
        ly = ((m(Xt).squeeze(-1) - yt) ** 2).mean()
        le = torch.nn.functional.binary_cross_entropy_with_logits(e(Xt).squeeze(-1), tt)
        opt.zero_grad()
        (ly + le).backward()
        opt.step()
    with torch.no_grad():
        ry = yt - m(Xt).squeeze(-1)
        rt = tt - torch.sigmoid(e(Xt).squeeze(-1))
    tau_net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
    )
    opt2 = torch.optim.Adam(tau_net.parameters(), lr=0.01)
    for _ in range(iters):
        r = (tau_net(Xt).squeeze(-1) * rt - ry) ** 2
        loss = r.mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        cate = tau_net(Xt).squeeze(-1).numpy()
        # naive: regress y on [X, t, t·X]
        D = np.concatenate([X, t[:, None], X * t[:, None]], 1)
        w = np.linalg.solve(D.T @ D + 0.1 * np.eye(D.shape[1]), D.T @ y)
        x1 = np.concatenate([X, np.ones((len(X), 1)), X], 1)
        x0 = np.concatenate([X, np.zeros((len(X), 1)), np.zeros(X.shape)], 1)
        naive = x1 @ w - x0 @ w
    return {
        "synthetic_rlearner_pehe": pehe(cate, tau),
        "synthetic_rlearner_naive_pehe": pehe(naive, tau),
        "synthetic_rlearner_pehe_gain": pehe(naive, tau) - pehe(cate, tau),
        "torch_available": 1.0,
    }
