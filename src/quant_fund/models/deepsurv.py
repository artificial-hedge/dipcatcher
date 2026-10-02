"""DeepSurv (Katzman et al. 2018) — neural Cox PH: MLP risk score
trained by negative partial log-likelihood vs linear Cox PH C-index.
"""

from __future__ import annotations

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deepsurv requires torch (pip install -e .[nn])") from exc
    return torch


def bench_deepsurv(seed: int = 2101, iters: int = 600) -> dict[str, float]:
    torch = _torch()
    X, t, e, beta = sv_data(seed)
    Xt, tt, et, _ = sv_data(seed + 1, n=200, beta=beta)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(6, 24), torch.nn.ELU(), torch.nn.Linear(24, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    Xt_ = torch.tensor(X).float()
    tt_ = torch.tensor(t).float()
    et_ = torch.tensor(e).float()
    for _ in range(iters):
        r = net(Xt_).squeeze(-1)
        ev_idx = torch.nonzero(et_ == 1).squeeze(-1)
        mask = (tt_[None, :] >= tt_[ev_idx][:, None]).float()
        log_risk = torch.logsumexp(r[None, :].expand(len(ev_idx), -1) + (1 - mask) * -1e9, dim=1)
        ll = -(r[ev_idx] - log_risk).mean()
        opt.zero_grad()
        ll.backward()
        opt.step()
    with torch.no_grad():
        rp = net(torch.tensor(Xt).float()).squeeze(-1).numpy()
    c_deep = cindex(tt, rp, et)
    b = cox_ph(X, t, e)
    c_cox = cindex(tt, Xt @ b, et)
    return {
        "synthetic_deepsurv_cindex": c_deep,
        "synthetic_cox_cindex": c_cox,
        "synthetic_deepsurv_gain": c_deep - c_cox,
        "torch_available": 1.0,
    }
