"""Conditional normalizing-flow regression — affine-coupling flow on y
conditioned on x (RealNVP-style, 1-D y): base N(0,1), condition net emits
log-scale/shift per coupling. Test log-density vs Gaussian baseline.
"""

from __future__ import annotations

from quant_fund.models._cd_synth import cd_data, gauss_logpdf


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("flow_regression requires torch (pip install -e .[nn])") from exc
    return torch


def bench_flow_regression(seed: int = 827, iters: int = 400, K: int = 4) -> dict[str, float]:
    torch = _torch()
    x, y, x_te, y_te = cd_data(seed)
    X = torch.tensor(x).float()
    Y = torch.tensor(y).float()
    Xt = torch.tensor(x_te).float()
    torch.manual_seed(seed)
    conds = torch.nn.ModuleList(
        [
            torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
            for _ in range(K)
        ]
    )
    opt = torch.optim.Adam(conds.parameters(), lr=0.005)
    base = torch.distributions.Normal(0.0, 1.0)

    def forward(xx, yy):
        z = yy
        ladj = torch.zeros(len(xx))
        for c in conds:
            st = c(xx)
            s, t = st[:, 0].clamp(-3, 3), st[:, 1]
            z = (z - t) * torch.exp(-s)
            ladj = ladj - s
        return z, ladj

    for _i in range(iters):
        z, ladj = forward(X, Y)
        loss = -(base.log_prob(z) + ladj).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z, ladj = forward(Xt, torch.tensor(y_te).float())
        ll_te = (base.log_prob(z) + ladj).numpy()
    ll_gauss = gauss_logpdf(y_te, float(y.mean()), float(y.std()))
    return {
        "synthetic_cnf_test_ll": float(ll_te.mean()),
        "synthetic_cnf_gauss_ll": float(ll_gauss.mean()),
        "synthetic_cnf_ll_gain": float(ll_te.mean() - ll_gauss.mean()),
        "torch_available": 1.0,
    }
