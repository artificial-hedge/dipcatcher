"""Contrastive divergence (Hinton 2002) — CD-k: positive phase from
data, negative phase from k-step Langevin initialized at data. EBM
loss = E_pos - E_neg; MMD vs Gaussian baseline.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, langevin, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("contrastive_divergence requires torch (pip install -e .[nn])") from exc
    return torch


def bench_contrastive_divergence(
    seed: int = 2441, iters: int = 500, k: int = 15
) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    net = make_energy(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))]
        xn = langevin(torch, net, 256, steps=k, step=0.05, seed=None, x0=x)
        loss = net(x).mean() - net(xn).mean() + 0.05 * (net(x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    x0 = torch.tensor(Xte).float()
    S = langevin(torch, net, 500, steps=60, step=0.05, seed=seed, x0=x0).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_cd_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_cd_gain": mb - m,
        "synthetic_torch_available": 1.0,
    }
