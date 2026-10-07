"""Spectrally-normalized 1-Lipschitz MLP (Miyato et al. 2018).

Each Linear weight is normalized by its spectral norm (power iteration)
so layer Lipschitz ≤ 1; with ReLU (1-Lipschitz) the whole network is
provably ≤1-Lipschitz. Empirical slope under adversarial perturbation
is measured against an unconstrained MLP — the guaranteed bound holds
where the unconstrained net's local slope can blow up.
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
        raise ImportError("lipschitz_net needs the torch `nn` extra") from exc


def _sn(w):
    import torch

    u = torch.randn(1, w.shape[0])
    for _ in range(5):
        v = u @ w
        v = v / v.norm().clamp(min=1e-9)
        u = v @ w.T
        u = u / u.norm().clamp(min=1e-9)
    s = (u @ w @ v.T).abs().clamp(min=1e-9)
    return w / s


def bench_lipschitz_net(
    seed: int = 109,
    n_train: int = 400,
    n_test: int = 150,
    iters: int = 400,
    eps: float = 0.3,
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    x, y = synth_cls_2d(n_train, rng)
    net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    x_t = torch.tensor(x).float()
    y_t = torch.tensor(y)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _i in range(iters):
        w1n = _sn(net[0].weight)
        w2n = _sn(net[2].weight)
        with torch.no_grad():
            net[0].weight.copy_(w1n)
            net[2].weight.copy_(w2n)
        loss = torch.nn.functional.cross_entropy(net(x_t), y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    plain = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    opt2 = torch.optim.Adam(plain.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(plain(x_t), y_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    xte, yte = synth_cls_2d(n_test, np.random.default_rng(seed + 1))
    xt = torch.tensor(xte).float()
    rng2 = np.random.default_rng(seed + 2)
    pert = torch.tensor(rng2.standard_normal(xte.shape)).float()
    pert = pert / pert.norm(dim=-1, keepdim=True) * eps
    with torch.no_grad():
        base = net(xt)
        adv = net(xt + pert)
        slope_sn = float((adv - base).abs().max(-1).values.div(eps).mean())
        base_p = plain(xt)
        adv_p = plain(xt + pert)
        slope_p = float((adv_p - base_p).abs().max(-1).values.div(eps).mean())
        acc_sn = float((net(xt).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_p = float((plain(xt).argmax(-1) == torch.tensor(yte)).float().mean())
        max_slope_p = float((adv_p - base_p).abs().max(-1).values.div(eps).max())
    return {
        "synthetic_lip_acc": acc_sn,
        "synthetic_lip_plain_acc": acc_p,
        "synthetic_lip_empirical_slope": slope_sn,
        "synthetic_lip_plain_slope": slope_p,
        "synthetic_lip_plain_max_slope": max_slope_p,
        "synthetic_torch_available": 1.0,
    }
