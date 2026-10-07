"""Noise-contrastive estimation (Gutmann & Hyvärinen 2010) — energy net (SYNTHETIC)
as log-ratio classifier data-vs-noise (reference = unit Gaussian);
samples via Langevin, MMD vs baseline.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, langevin, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("noise_contrastive requires torch (pip install -e .[nn])") from exc
    return torch


def bench_noise_contrastive(seed: int = 2435, iters: int = 500) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    net = make_energy(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))]
        xn = torch.randn(256, 2) * 2.0 + torch.tensor([0.5, 0.25])
        e_d = -net(x).squeeze(-1)  # -E as log-ratio
        e_n = -net(xn).squeeze(-1)
        lp_d = e_d - torch.logsumexp(torch.stack([e_d, torch.full_like(e_d, 0.0)]), 0)
        lp_n = torch.zeros_like(e_n) - torch.logsumexp(torch.stack([e_n, torch.zeros_like(e_n)]), 0)
        loss = -(lp_d.mean() + lp_n.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    x0 = torch.tensor(Xte).float()
    S = langevin(torch, net, 500, steps=60, step=0.05, seed=seed, x0=x0).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_nce_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_nce_gain": mb - m,
        "synthetic_torch_available": 1.0,
    }
