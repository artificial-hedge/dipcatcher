"""DRSA (Ren et al. 2019) — deep recurrent survival analysis: LSTM (SYNTHETIC)
over time slices outputs per-interval event probabilities; C-index vs
Cox.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("drsa_surv requires torch (pip install -e .[nn])") from exc
    return torch


def bench_drsa_surv(seed: int = 2127, iters: int = 500, K: int = 10) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    cell = torch.nn.LSTMCell(6, 24)
    head = torch.nn.Linear(24, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(head.parameters()), lr=0.005)
    edges = np.quantile(t, np.linspace(0, 1, K + 1))
    edges[0] = 0
    tb = np.clip(np.searchsorted(edges, t) - 1, 0, K - 1)
    Xt_ = torch.tensor(X).float()
    tb_ = torch.tensor(tb)
    et_ = torch.tensor(e).float()

    def seq_probs(x):
        h = torch.zeros(len(x), 24)
        c = torch.zeros(len(x), 24)
        probs = []
        for _k in range(K):
            h, c = cell(x, (h, c))
            probs.append(torch.sigmoid(head(h).squeeze(-1)))
        return torch.stack(probs, -1)

    for _ in range(iters):
        hz = seq_probs(Xt_)
        pmf = hz * torch.cat([torch.ones(len(X), 1), torch.cumprod(1 - hz, -1)[:, :-1]], -1)
        pmf_ev = pmf.gather(1, tb_[:, None]).squeeze(-1).clamp_min(1e-7)
        surv = torch.cumprod(1 - hz, -1).gather(1, tb_[:, None]).squeeze(-1).clamp_min(1e-7)
        ll = -(et_ * torch.log(pmf_ev) + (1 - et_) * torch.log(surv)).mean()
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        hz = seq_probs(torch.tensor(Xt).float()).numpy()
    risk = (hz * np.arange(1, K + 1)).sum(-1)
    c_dr = cindex(tt, risk, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_drsa_cindex": c_dr,
        "synthetic_cox_cindex": c_cox,
        "synthetic_drsa_gain": c_dr - c_cox,
        "synthetic_torch_available": 1.0,
    }
