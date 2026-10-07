"""Neural Spline Flow (Durkan et al. 2019) — monotonic rational-quadratic
coupling: x_b transformed by an RQ spline whose knots/slopes come from
x_a via a small net. NLL vs Gaussian on pinwheel.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import gauss_nll, pinwheel


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("neural_spline_flow requires torch (pip install -e .[nn])") from exc
    return torch


def _rqs(x, w, h, d, B=3.0):
    """Monotonic RQ spline on [-B,B]; x,w,h,d (N,K...) — outside: identity."""
    import torch

    K = w.shape[-1]
    w = torch.nn.functional.softmax(w, -1) * 2 * B
    h = torch.nn.functional.softmax(h, -1) * 2 * B
    d = torch.nn.functional.softplus(d)
    xk = torch.cumsum(w, -1) - B
    yk = torch.cumsum(h, -1) - B
    xk = torch.cat([torch.full_like(xk[..., :1], -B), xk], -1)
    yk = torch.cat([torch.full_like(yk[..., :1], -B), yk], -1)
    dk = torch.cat([torch.ones_like(d[..., :1]), d, torch.ones_like(d[..., :1])], -1)
    inside = ((x >= -B) & (x <= B)).squeeze(-1)
    xi = torch.clamp(x, -B + 1e-6, B - 1e-6).squeeze(-1)
    idx = torch.searchsorted(xk, xi.unsqueeze(-1)).squeeze(-1) - 1
    idx = idx.clamp(0, K - 1)

    def g(t):
        return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    x0, x1 = g(xk), g(xk[..., 1:])
    y0, y1 = g(yk), g(yk[..., 1:])
    d0, d1 = g(dk), g(dk[..., 1:])
    s = (y1 - y0) / (x1 - x0)
    th = (xi - x0) / (x1 - x0)
    num = s * th * th + d0 * th * (1 - th)
    den = s + (d0 + d1 - 2 * s) * th * (1 - th)
    y = y0 + (y1 - y0) * num / den
    dydx = s * s * (d1 * th * th + 2 * s * th * (1 - th) + d0 * (1 - th) ** 2) / (den * den)
    return (
        torch.where(inside, y, xi).unsqueeze(-1),
        torch.where(inside, dydx, torch.ones_like(dydx)).unsqueeze(-1),
    )


def bench_neural_spline_flow(seed: int = 2293, iters: int = 600, K: int = 8) -> dict[str, float]:
    torch = _torch()
    Xtr = pinwheel(seed)
    Xte = pinwheel(seed + 1, n=500)
    torch.manual_seed(seed)
    blocks = 3
    nets = [
        torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3 * K))
        for _ in range(blocks)
    ]
    params = [p for m in nets for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.003)
    Xt = torch.tensor(Xtr).float()

    def fwd(x):
        ld = 0.0
        for m in nets:
            xa, xb = x[:, :1], x[:, 1:]
            prm = m(xa)
            w, h, d = prm[:, :K], prm[:, K : 2 * K], prm[:, 2 * K :]
            xb, dydx = _rqs(xb, w, h, d)
            ld = ld + torch.log(dydx + 1e-9).squeeze(-1)
            x = torch.cat([xb, xa], -1)
        return x, ld

    for _ in range(iters):
        z, ld = fwd(Xt)
        nll = (0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    with torch.no_grad():
        z, ld = fwd(torch.tensor(Xte).float())
        nll_te = float((0.5 * (z**2).sum(-1) + np.log(2 * np.pi) - ld).mean())
    base = gauss_nll(Xtr, Xte)
    return {
        "synthetic_nsf_nll": nll_te,
        "synthetic_gauss_nll": base,
        "synthetic_nsf_gain": base - nll_te,
        "synthetic_torch_available": 1.0,
    }
