"""Implicit score matching (Hyvärinen 2005) — train E(x) so that (SYNTHETIC)
∇E ≈ -∇ log p_data via tr(∇²E) + 0.5|∇E|²; Langevin samples then
MMD-evaluated vs moons data.
"""

from __future__ import annotations

from quant_fund.models._eb_synth import gauss_baseline_mmd, langevin, make_energy, mmd, moon_data


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("score_matching requires torch (pip install -e .[nn])") from exc
    return torch


def bench_score_matching(seed: int = 2423, iters: int = 400) -> dict[str, float]:
    torch = _torch()
    Xtr, Xte = moon_data(seed)
    torch.manual_seed(seed)
    net = make_energy(torch)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    Xt = torch.tensor(Xtr).float()
    for _ in range(iters):
        x = Xt[torch.randint(0, len(Xt), (256,))].requires_grad_(True)
        e = net(x).sum()
        g = torch.autograd.grad(e, x, create_graph=True)[0]
        div = 0.0
        for j in range(2):
            div = div + torch.autograd.grad(g[:, j].sum(), x, create_graph=True)[0][:, j]
        loss = (div + 0.5 * (g * g).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    x0 = torch.tensor(Xte).float()
    S = langevin(torch, net, 500, steps=60, step=0.05, seed=seed, x0=x0).numpy()
    m = mmd(S, Xte)
    mb = gauss_baseline_mmd(Xtr, Xte)
    return {
        "synthetic_sm_mmd": m,
        "synthetic_gauss_mmd": mb,
        "synthetic_sm_gain": mb - m,
        "synthetic_torch_available": 1.0,
    }
