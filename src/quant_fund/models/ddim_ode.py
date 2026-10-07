"""DDIM ODE sampler (Song et al. 2021) — deterministic eta=0 DDIM on a (SYNTHETIC)
preconditioned epsilon-net; 20-step quality vs 20-step ancestral DDPM.
MMD comparison.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dfx_synth import _mmd, gauss_mmd, synth_regime_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ddim_ode requires torch (pip install -e .[nn])") from exc
    return torch


def bench_ddim_ode(seed: int = 1517, iters: int = 900, steps: int = 20) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, _ = synth_regime_windows(400, 16, rng)
    Xtr, Xte = torch.tensor(X[:300]).float(), X[300:].astype(np.float64)
    net = torch.nn.Sequential(torch.nn.Linear(16 + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    T = 200
    betas = torch.linspace(1e-4, 0.05, T)
    ab = torch.cumprod(1 - betas, 0)
    for _ in range(iters):
        idx = rng.integers(0, len(Xtr), 64)
        x0 = Xtr[idx]
        t = torch.randint(0, T, (64, 1)).float() / T
        tt = (t[:, 0] * T).long().clamp(0, T - 1)
        eps = torch.randn_like(x0)
        xt = ab[tt].sqrt()[:, None] * x0 + (1 - ab[tt]).sqrt()[:, None] * eps
        loss = ((net(torch.cat([xt, t], 1)) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    tgrid = np.linspace(T - 1, 0, steps).astype(int)
    xs_d = torch.randn(300, 16)
    xs_p = xs_d.clone()
    with torch.no_grad():
        for i in range(steps - 1):
            t_cur, t_next = int(tgrid[i]), int(tgrid[i + 1])
            a_c, a_n = ab[t_cur], ab[t_next]
            tt = torch.full((300, 1), t_cur / T)
            eps_d = net(torch.cat([xs_d, tt], 1))
            xs_d = (
                a_n.sqrt() * (xs_d - (1 - a_c).sqrt() * eps_d) / a_c.sqrt()
                + (1 - a_n).sqrt() * eps_d
            )
            eps_p = net(torch.cat([xs_p, tt], 1))
            sig = 0.25 * ((1 - a_n) / (1 - a_c)).sqrt() * (1 - a_c / a_n).clamp_min(0).sqrt()
            xs_p = (
                a_n.sqrt() * (xs_p - (1 - a_c).sqrt() * eps_p) / a_c.sqrt()
                + (1 - a_n).sqrt() * eps_p
                + sig * torch.randn_like(xs_p)
            )
    m_d = _mmd(xs_d.numpy().astype(np.float64), Xte, bw=1.0)
    m_p = _mmd(xs_p.numpy().astype(np.float64), Xte, bw=1.0)
    g = gauss_mmd(np.random.default_rng(seed + 1), Xte)
    return {
        "synthetic_ddim_mmd": m_d,
        "synthetic_ddim_ddpm_mmd": m_p,
        "synthetic_ddim_gauss_mmd": g,
        "synthetic_ddim_mmd_gain_vs_ddpm": m_p - m_d,
        "synthetic_torch_available": 1.0,
    }
