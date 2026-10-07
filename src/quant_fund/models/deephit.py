"""DeepHit (Lee et al. 2018) — discrete-time competing-risk network:
softmax over time-bin pmf + event indicators trained by likelihood on
discretized (time, event) vs Cox PH C-index.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deephit requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deephit(seed: int = 2107, iters: int = 500, K: int = 12) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, K))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    edges = np.quantile(t, np.linspace(0, 1, K + 1))
    edges[0] = 0
    tb = np.clip(np.searchsorted(edges, t) - 1, 0, K - 1)
    Xt_ = torch.tensor(X).float()
    tb_ = torch.tensor(tb)
    et_ = torch.tensor(e).float()
    for _ in range(iters):
        logits = net(Xt_)
        pmf = torch.softmax(logits, -1)
        # event likelihood: pmf at tb; censored: survival past tb
        pmf_ev = pmf.gather(1, tb_[:, None]).squeeze(-1)
        surv = (1 - torch.cumsum(pmf, -1).gather(1, tb_[:, None]).squeeze(-1) + pmf_ev).clamp_min(
            1e-7
        )
        ll = -(et_ * torch.log(pmf_ev + 1e-7) + (1 - et_) * torch.log(surv)).mean()
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        pmf = torch.softmax(net(torch.tensor(Xt).float()), -1).numpy()
    risk = -(pmf @ np.arange(K))  # earlier expected time → higher risk
    c_dh = cindex(tt, risk, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_deephit_cindex": c_dh,
        "synthetic_cox_cindex": c_cox,
        "synthetic_deephit_gain": c_dh - c_cox,
        "synthetic_torch_available": 1.0,
    }
