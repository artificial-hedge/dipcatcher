"""α-CROWN-lite linear relaxation bounds (Zhang et al. 2018).

For an unstable ReLU neuron (l < 0 < u), the linear relaxation
λ·z ≤ ReLU(z) ≤ α·(z − l) with α = u/(u−l) propagates strictly tighter
bounds than IBP's box. We lower-bound the true-class logit margin of a
2-layer MLP over the ε-ball and compare the certified fraction to IBP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._cert_synth import synth_cls_2d

FloatArray = NDArray[np.float64]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("crown_bound needs the torch `nn` extra") from exc


def bench_crown_bound(
    seed: int = 107,
    n_train: int = 400,
    n_test: int = 150,
    eps: float = 0.35,
    iters: int = 400,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_cls_2d(n_train, rng)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    xte, yte = synth_cls_2d(n_test, np.random.default_rng(seed + 1))
    xt = torch.tensor(xte).float()
    yt = torch.tensor(yte)
    w1, b1 = net[0].weight, net[0].bias
    w2, b2 = net[2].weight, net[2].bias

    # IBP bounds on pre-activations z = W1 x + b1 over the ε-ball
    c1 = xt @ w1.T + b1
    r1 = eps * w1.abs().sum(-1)[None, :] * torch.ones_like(c1)
    lo1, hi1 = c1 - r1, c1 + r1

    # margin logit: w_margin = w2[y]−w2[1−y] applied to h = ReLU(z)
    eye = torch.nn.functional.one_hot(yt, 2).float()
    d = (w2[1][None, :] - w2[0][None, :]) * (2 * eye[:, 1:] - 1)
    d = torch.where(yt[:, None] == 1, w2[1] - w2[0], w2[0] - w2[1])
    db = torch.where(yt == 1, b2[1] - b2[0], b2[0] - b2[1])

    unstable = (lo1 < 0) & (hi1 > 0)
    alpha = torch.where(unstable, hi1 / (hi1 - lo1).clamp(min=1e-9), torch.zeros_like(hi1))
    lam = torch.ones_like(hi1)
    coef = torch.where(
        lo1 >= 0,
        torch.ones_like(lo1),
        torch.where(
            hi1 <= 0,
            torch.zeros_like(hi1),
            torch.where(d > 0, lam, alpha),
        ),
    )
    const = torch.where(unstable & (d < 0), -d * alpha * lo1, torch.zeros_like(lo1))
    slope = torch.einsum("nh,hd->nd", d * coef, w1)
    lb = (
        db
        + (d * coef * b1[None, :]).sum(-1)
        + const.sum(-1)
        + (slope * xt).sum(-1)
        - eps * slope.abs().sum(-1)
    )
    crown_cert = float((lb > 0).float().mean())

    lo1r, hi1r = torch.clamp(lo1, min=0), torch.clamp(hi1, min=0)
    c2 = (lo1r + hi1r) / 2 @ w2.T + b2
    r2 = (hi1r - lo1r) / 2 @ w2.abs().T
    lo2, hi2 = c2 - r2, c2 + r2
    margin_ibp = torch.where(yt == 1, lo2[:, 1] - hi2[:, 0], lo2[:, 0] - hi2[:, 1])
    ibp_cert = float((margin_ibp > 0).float().mean())
    return {
        "synthetic_crown_certified_acc": crown_cert,
        "synthetic_crown_ibp_certified_acc": ibp_cert,
        "synthetic_crown_gain": crown_cert - ibp_cert,
        "synthetic_crown_mean_lb": float(lb.mean()),
        "torch_available": 1.0,
    }
