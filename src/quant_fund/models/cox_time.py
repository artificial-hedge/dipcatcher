"""Cox-Time (Kvamme et al. 2019) — Cox model where the risk function
also sees time: net(x, t) trained by case-control partial likelihood;
non-PH fixtures gain over time-constant Cox.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cox_time requires torch (pip install -e .[nn])") from exc
    return torch


def bench_cox_time(seed: int = 2113, iters: int = 500) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(7, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    Xt_ = torch.tensor(X).float()
    tt_ = torch.tensor(t).float()
    et_ = torch.tensor(e).float()
    for _ in range(iters):
        # case-control: each event vs its risk set sample
        ev = torch.nonzero(et_ == 1).squeeze(-1)
        ll = net(torch.zeros(1, 7)).sum() * 0.0
        n_ev = len(ev)
        for i in ev[:64]:
            rs = torch.nonzero(tt_ >= tt_[i]).squeeze(-1)
            x_i = torch.cat([Xt_[i], tt_[i][None]])
            r_i = net(x_i).sum()
            rs_sub = rs[torch.randint(0, len(rs), (min(32, len(rs)),))]
            xj = torch.cat([Xt_[rs_sub], tt_[i].expand(len(rs_sub))[:, None]], -1)
            ll = ll + r_i - torch.logsumexp(net(xj).squeeze(-1), 0)
        ll = -ll / min(n_ev, 64)
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        xq = torch.cat(
            [torch.tensor(Xt).float(), torch.full((len(Xt), 1), float(np.median(tt)))], -1
        )
        rp = net(xq).squeeze(-1).numpy()
    c_ct = cindex(tt, rp, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_coxtime_cindex": c_ct,
        "synthetic_cox_cindex": c_cox,
        "synthetic_coxtime_gain": c_ct - c_cox,
        "torch_available": 1.0,
    }
