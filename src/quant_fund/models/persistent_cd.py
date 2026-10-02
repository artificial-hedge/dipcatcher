"""Persistent CD (Tieleman 2008) — negative phase continues a
persistent Langevin chain across updates (fantasy particles), better
mode coverage than CD-k; MMD vs Gaussian baseline.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, langevin, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("persistent_cd requires torch (pip install -e .[nn])") from exc
    return torch


def bench_persistent_cd(seed: int = 2447, iters: int = 500, k: int = 8) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    net = make_energy(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    chain = torch.randn(256, 2) * 2  # persistent fantasy particles
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))]
        chain = langevin(torch, net, 256, steps=k, step=0.05, seed=None, x0=chain)
        loss = net(x).mean() - net(chain).mean() + 0.05 * (net(x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if torch.rand(1) < 0.05:
            chain = torch.randn(256, 2) * 2  # occasional restart
    x0 = torch.tensor(Xte).float()
    S = langevin(torch, net, 500, steps=60, step=0.05, seed=seed, x0=x0).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_pcd_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_pcd_gain": mb - m,
        "torch_available": 1.0,
    }
