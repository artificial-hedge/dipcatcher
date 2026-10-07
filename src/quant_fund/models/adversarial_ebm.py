"""Adversarial EBM / cooperative nets (Xie et al. 2016) — generator
proposal + energy critic trained adversarially (the energy net scores
real-vs-generated like a WGAN critic); MMD vs Gaussian baseline.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("adversarial_ebm requires torch (pip install -e .[nn])") from exc
    return torch


def bench_adversarial_ebm(seed: int = 2453, iters: int = 500) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    gen = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2))
    net = make_energy(torch)  # critic: low energy = real
    og = torch.optim.Adam(gen.parameters(), lr=0.003)
    oc = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))]
        z = torch.randn(256, 4)
        xg = gen(z)
        lc = -net(x).mean() + net(xg).mean() + 0.1 * ((net(xg) ** 2).mean() + (net(x) ** 2).mean())
        oc.zero_grad()
        lc.backward()
        oc.step()
        z = torch.randn(256, 4)
        lg = -net(gen(z)).mean()  # gen wants low energy
        og.zero_grad()
        lg.backward()
        og.step()
    with torch.no_grad():
        S = gen(torch.randn(500, 4)).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_aebm_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_aebm_gain": mb - m,
        "synthetic_torch_available": 1.0,
    }
