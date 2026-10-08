"""Cold diffusion (Bansal et al. 2022) — non-noise degradation: blur/ (SYNTHETIC)
downsample degradation operator D(x,t) = blur(x, sigma_t); train
restoration net R(x_t,t)≈x0; sample by iterative restore. MMD vs
noise-only baseline.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._dfx_synth import _mmd, gauss_mmd, synth_regime_windows


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("cold_diffusion requires torch (pip install -e .[nn])") from exc
    return torch


def _blur(x: np.ndarray, t: float) -> np.ndarray:
    # moving-average blur with kernel width ~ 1 + 3t
    w = int(1 + 3 * t)
    kern = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    out = np.apply_along_axis(lambda r: np.convolve(r, kern, "valid"), 1, xp)
    return out[:, : x.shape[1]]


def bench_cold_diffusion(seed: int = 1523, iters: int = 900, steps: int = 20) -> dict[str, float]:
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X, _ = synth_regime_windows(400, 16, rng)
    Xtr, Xte = torch.tensor(X[:300]).float(), X[300:].astype(np.float64)
    net = torch.nn.Sequential(torch.nn.Linear(16 + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(iters):
        idx = rng.integers(0, len(Xtr), 64)
        x0 = Xtr[idx].numpy()
        t = rng.uniform(0, 1)
        xt = torch.tensor(_blur(x0, t)).float()
        tt = torch.full((64, 1), t)
        loss = ((net(torch.cat([xt, tt], 1)) - Xtr[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    xs = torch.tensor(
        _blur(np.tile(Xtr.numpy().mean(0), (300, 1)) + 0.05 * rng.standard_normal((300, 16)), 1.0)
    ).float()
    with torch.no_grad():
        for i in range(steps - 1, -1, -1):
            t = (i + 1) / steps
            tt = torch.full((300, 1), t)
            x0_hat = net(torch.cat([xs, tt], 1))
            # iterative update: x_{t-1} = x_t - D(x0_hat,t) + D(x0_hat,t-1)
            dt = np.asarray(_blur(x0_hat.numpy(), t)) - np.asarray(
                _blur(x0_hat.numpy(), (i) / steps)
            )
            xs = xs - torch.tensor(dt).float()
    gen = xs.numpy().astype(np.float64)
    m = _mmd(gen, Xte, bw=1.0)
    g = gauss_mmd(np.random.default_rng(seed + 1), Xte)
    return {
        "synthetic_cold_mmd": m,
        "synthetic_cold_gauss_mmd": g,
        "synthetic_cold_mmd_gain": g - m,
        "synthetic_torch_available": 1.0,
    }
