"""Stochastic interpolant (Albergo & Vanden-Eijnden 2023) — velocity
field on x_t = (1-t)x0 + t x1 + gamma·sqrt(t(1-t)) z; Euler sampler.
MMD vs rectified-flow-style ODE baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dfx_synth import _mmd, gauss_mmd, synth_regime_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("stoch_interp requires torch (pip install -e .[nn])") from exc
    return torch


def bench_stoch_interp(
    seed: int = 1511, iters: int = 900, steps: int = 24, gamma: float = 0.3
) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, _ = synth_regime_windows(400, 16, rng)
    Xtr, Xte = torch.tensor(X[:300]).float(), X[300:].astype(np.float64)
    net = torch.nn.Sequential(torch.nn.Linear(16 + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(iters):
        idx = rng.integers(0, len(Xtr), 64)
        x0, x1 = torch.randn(64, 16), Xtr[idx]
        t = torch.rand(64, 1)
        z = torch.randn_like(x0)
        xt = (1 - t) * x0 + t * x1 + gamma * (t * (1 - t)).sqrt() * z
        # target: dx/dt = x1 - x0 + gamma*(1-2t)/(2*sqrt(t(1-t))) z
        tgt = x1 - x0 + gamma * (1 - 2 * t) / (2 * (t * (1 - t) + 1e-6).sqrt()) * z
        loss = ((net(torch.cat([xt, t], 1)) - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    xs = torch.randn(300, 16)
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((300, 1), i / steps)
            xs = xs + net(torch.cat([xs, t], 1)) / steps
    gen = xs.numpy().astype(np.float64)
    m = _mmd(gen, Xte, bw=1.0)
    g = gauss_mmd(np.random.default_rng(seed + 1), Xte)
    return {
        "synthetic_si_mmd": m,
        "synthetic_si_gauss_mmd": g,
        "synthetic_si_mmd_gain": g - m,
        "synthetic_torch_available": 1.0,
    }
