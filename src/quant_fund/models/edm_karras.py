"""EDM (Karras et al. 2022) — elucidated diffusion: sigma-conditional (SYNTHETIC)
denoiser with Karras preconditioning (c_skip/c_out/c_noise) and
Heun 2nd-order sampler. MMD vs Gauss baseline on regime windows.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dfx_synth import _mmd, gauss_mmd, synth_regime_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("edm_karras requires torch (pip install -e .[nn])") from exc
    return torch


def bench_edm_karras(seed: int = 1501, iters: int = 900, steps: int = 18) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, _ = synth_regime_windows(400, 16, rng)
    Xtr, Xte = torch.tensor(X[:300]).float(), X[300:].astype(np.float64)
    sd = float(Xtr.std())
    net = torch.nn.Sequential(
        torch.nn.Linear(16 + 1, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 16),
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sd_data = max(sd, 1e-6)

    def denoise(x, sig):
        c_skip = sd_data**2 / (sig**2 + sd_data**2)
        c_out = sig * sd_data / (sig**2 + sd_data**2).sqrt()
        c_in = 1 / (sig**2 + sd_data**2).sqrt()
        c_noise = 0.25 * sig.log()
        return c_skip * x + c_out * net(torch.cat([c_in * x, c_noise.expand(len(x), 1)], 1))

    for _ in range(iters):
        idx = rng.integers(0, len(Xtr), 64)
        x0 = Xtr[idx]
        sig = torch.tensor(np.exp(rng.uniform(np.log(0.02), np.log(5.0), (64, 1)))).float()
        eps = torch.randn_like(x0)
        loss = (((denoise(x0 + sig * eps, sig) - x0) / sig.clamp_min(1e-4)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Heun sampler
    sigs = np.geomspace(5.0, 0.02, steps)
    xs = torch.randn(300, 16) * sigs[0]
    with torch.no_grad():
        for i in range(steps - 1):
            s_cur, s_next = float(sigs[i]), float(sigs[i + 1])
            d = (xs - denoise(xs, torch.tensor(s_cur))) / s_cur
            x_next = xs + (s_next - s_cur) * d
            if s_next > 1e-3:
                d2 = (x_next - denoise(x_next, torch.tensor(s_next))) / s_next
                x_next = xs + (s_next - s_cur) * (d + d2) / 2
            xs = x_next
    gen = xs.numpy().astype(np.float64)
    m = _mmd(gen, Xte, bw=1.0)
    g = gauss_mmd(np.random.default_rng(seed + 1), Xte)
    return {
        "synthetic_edm_mmd": m,
        "synthetic_edm_gauss_mmd": g,
        "synthetic_edm_mmd_gain": g - m,
        "synthetic_torch_available": 1.0,
    }
