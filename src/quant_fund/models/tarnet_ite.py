"""TARNet — two-head ITE network (Shalit et al. 2017).

Shared trunk + treatment-specific heads trained on the factual arm
only; ITE = head1(x) − head0(x). PEHE vs naive single-regressor ITE.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._causal_synth import synth_observational


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("tarnet_ite requires torch (pip install -e .[nn])") from exc
    return torch


def bench_tarnet_ite(
    seed: int = 557,
    n: int = 800,
    iters: int = 120,
    h: int = 24,
) -> dict[str, float]:
    torch = _torch()
    x, t, y, tau, _e = synth_observational(seed, n)
    half = n // 2
    x_t = torch.tensor(x[:half]).float()
    t_t = torch.tensor(t[:half]).float()
    y_t = torch.tensor(y[:half]).float()
    x_te = torch.tensor(x[half:]).float()
    tau_te = tau[half:]
    torch.manual_seed(seed)
    trunk = torch.nn.Sequential(torch.nn.Linear(5, h), torch.nn.ReLU())
    head0 = torch.nn.Linear(h, 1)
    head1 = torch.nn.Linear(h, 1)
    opt = torch.optim.Adam(
        list(trunk.parameters()) + list(head0.parameters()) + list(head1.parameters()), lr=0.01
    )
    for _i in range(iters):
        z = trunk(x_t)
        pred = head1(z).squeeze(-1) * t_t + head0(z).squeeze(-1) * (1 - t_t)
        loss = ((pred - y_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ite = (head1(trunk(x_te)) - head0(trunk(x_te))).squeeze(-1).numpy()
    pehe_t = float(np.sqrt(np.mean((ite - tau_te) ** 2)))
    # naive single regressor with t as feature
    torch.manual_seed(seed)
    xt = torch.cat([x_t, t_t.unsqueeze(1)], 1)
    xt_te0 = torch.cat([x_te, torch.zeros(len(x_te), 1)], 1)
    xt_te1 = torch.cat([x_te, torch.ones(len(x_te), 1)], 1)
    nv = torch.nn.Sequential(torch.nn.Linear(6, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
    opt = torch.optim.Adam(nv.parameters(), lr=0.01)
    for _i in range(iters):
        loss = ((nv(xt).squeeze(-1) - y_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ite_n = (nv(xt_te1) - nv(xt_te0)).squeeze(-1).numpy()
    pehe_n = float(np.sqrt(np.mean((ite_n - tau_te) ** 2)))
    return {
        "synthetic_tarnet_pehe": pehe_t,
        "synthetic_tarnet_naive_pehe": pehe_n,
        "synthetic_tarnet_gain": pehe_n - pehe_t,
        "synthetic_tarnet_ate_err": float(abs(ite.mean() - tau_te.mean())),
        "synthetic_torch_available": 1.0,
    }
