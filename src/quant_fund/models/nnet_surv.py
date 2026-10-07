"""Nnet-survival (Gensheimer & Narasimhan 2019) — discrete-time neural
survival: per-interval hazard logits (logistic discrete hazard) vs Cox
PH C-index.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("nnet_surv requires torch (pip install -e .[nn])") from exc
    return torch


def bench_nnet_surv(seed: int = 2119, iters: int = 500, K: int = 10) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 24), torch.nn.ReLU(), torch.nn.Linear(24, K))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    edges = np.quantile(t, np.linspace(0, 1, K + 1))
    edges[0] = 0
    tb = np.clip(np.searchsorted(edges, t) - 1, 0, K - 1)
    Xt_ = torch.tensor(X).float()
    tb_ = torch.tensor(tb)
    et_ = torch.tensor(e).float()
    for _ in range(iters):
        logits = net(Xt_)
        hz = torch.sigmoid(logits)  # per-interval hazard
        survive = torch.cumprod(1 - hz, -1)
        pmf = hz * torch.cat([torch.ones(len(X), 1), survive[:, :-1]], -1)
        pmf_ev = pmf.gather(1, tb_[:, None]).squeeze(-1).clamp_min(1e-7)
        surv_past = survive.gather(1, tb_[:, None]).squeeze(-1).clamp_min(1e-7)
        ll = -(et_ * torch.log(pmf_ev) + (1 - et_) * torch.log(surv_past)).mean()
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        hz = torch.sigmoid(net(torch.tensor(Xt).float())).numpy()
    risk = (hz * np.arange(1, K + 1)).sum(-1)  # expected hazard mass
    c_ns = cindex(tt, risk, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_nnet_cindex": c_ns,
        "synthetic_cox_cindex": c_cox,
        "synthetic_nnet_gain": c_ns - c_cox,
        "synthetic_torch_available": 1.0,
    }
