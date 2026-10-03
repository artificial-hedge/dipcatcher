"""Snapshot ensembles (Huang et al. 2017) — cyclic LR: restart at high LR
every `cycle` steps, snapshot the weights at each minimum; ensemble
prediction = mixture mean/var.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._bdl_synth import bdl_data, coverage, nll_gauss


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("snapshot_ens requires torch (pip install -e .[nn])") from exc
    return torch


def bench_snapshot_ens(seed: int = 799, cycles: int = 4, cycle_len: int = 60) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te, x_ood = bdl_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()
    Xt = torch.tensor(x_te).float()
    Xo = torch.tensor(x_ood).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    snaps: list[dict] = []
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cycle_len)
    for _c in range(cycles):
        for _i in range(cycle_len):
            loss = ((net(X).squeeze(-1) - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        snaps.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        for g in opt.param_groups:
            g["lr"] = 0.05
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cycle_len)
    pt, po = [], []
    with torch.no_grad():
        for s in snaps:
            net.load_state_dict(s)
            pt.append(net(Xt).squeeze(-1).numpy())
            po.append(net(Xo).squeeze(-1).numpy())
    P = np.stack(pt)
    Po = np.stack(po)
    mu, var = P.mean(0), P.var(0) + 0.05
    return {
        "synthetic_se_nll": nll_gauss(y_te, mu, var),
        "synthetic_se_cov95": coverage(y_te, mu, np.sqrt(var)),
        "synthetic_se_ood_gap": float(Po.var(0).mean() / (P.var(0).mean() + 1e-9)),
        "synthetic_se_n_members": float(len(snaps)),
    }
