"""Denoising score matching (Vincent 2011) — perturb x with noise σ, (SYNTHETIC)
match ∇E to -(x̃-x)/σ². No Hessian needed; MMD vs Gaussian baseline.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, langevin, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("denoising_sm requires torch (pip install -e .[nn])") from exc
    return torch


def bench_denoising_sm(seed: int = 2429, iters: int = 500, sig: float = 0.15) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    net = make_energy(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))]
        xt = x + sig * torch.randn_like(x)
        xt = xt.requires_grad_(True)
        e = net(xt).sum()
        g = torch.autograd.grad(e, xt)[0]
        target = -(xt - x) / sig**2
        loss = ((g - target) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    x0 = torch.tensor(Xte).float()
    S = langevin(torch, net, 500, steps=60, step=0.05, seed=seed, x0=x0).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_dsm_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_dsm_gain": mb - m,
        "synthetic_torch_available": 1.0,
    }
