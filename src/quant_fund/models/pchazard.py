"""PC-Hazard (Kvamme & Borgan 2021) — piecewise-constant hazard neural
net: hazard rate per time bin via softplus head; survival = product of
exp(-hazard * bin-width). C-index vs Cox.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("pchazard requires torch (pip install -e .[nn])") from exc
    return torch


def bench_pchazard(seed: int = 2133, iters: int = 500, K: int = 10) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 24), torch.nn.ReLU(), torch.nn.Linear(24, K))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    edges = np.quantile(t, np.linspace(0, 1, K + 1))
    edges[0] = 1e-6
    tb = np.clip(np.searchsorted(edges, t) - 1, 0, K - 1)
    Xt_ = torch.tensor(X).float()
    tb_ = torch.tensor(tb)
    et_ = torch.tensor(e).float()
    widths = np.diff(edges)
    for _ in range(iters):
        hz = torch.nn.functional.softplus(net(Xt_))
        wt = torch.tensor(widths).float()
        cum_h = torch.cumsum(hz * wt, -1)
        hazard_at = hz.gather(1, tb_[:, None]).squeeze(-1).clamp_min(1e-7)
        cum_at = cum_h.gather(1, tb_[:, None]).squeeze(-1)
        ll = -(et_ * torch.log(hazard_at) - cum_at).mean()
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        hz = torch.nn.functional.softplus(net(torch.tensor(Xt).float())).numpy()
    risk = hz.mean(-1)  # mean hazard as risk score
    c_pc = cindex(tt, risk, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_pchazard_cindex": c_pc,
        "synthetic_cox_cindex": c_cox,
        "synthetic_pchazard_gain": c_pc - c_cox,
        "synthetic_torch_available": 1.0,
    }
